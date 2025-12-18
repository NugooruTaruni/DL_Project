#!/usr/bin/env python3
"""
train_vicreg.py

VicReg (Variance-Invariance-Covariance Regularization) with ConvNeXt
- Backbone: ConvNeXt-Base (~88M params)
- Method: Self-supervised learning with VicReg loss
- Loss: Variance + Invariance + Covariance
- Features: torch.compile, bfloat16, distributed training, LARS optimizer
- Dataset: CC3M only

Reference: "VicReg: Variance-Invariance-Covariance Regularization for Self-Supervised Learning"
"""

import argparse, math, random, time, json, os
from pathlib import Path
from typing import List
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms, models
from tqdm import tqdm

# --------------------------
# Config defaults
# --------------------------
DEFAULTS = {
    "ckpt_dir": "./ckpts_vicreg",
    "device": "cuda",
    "num_workers": 8,
    "batch_size": 256,
    "epochs": 200,
    "backbone": "convnext_base",
    "base_lr": 0.3,  # LARS works well with large LR
    "weight_decay": 1e-6,
    "warmup_epochs": 10,
    "print_every": 200,
    "seed": 42,
    "compile": True,
    "use_bfloat16": True,
    "grad_clip": 3.0,
    # VicReg specific
    "sim_coeff": 25.0,
    "std_coeff": 25.0,
    "cov_coeff": 1.0,
    "proj_dim": 8192,
}

# --------------------------
# Dataset Path
# --------------------------
CC3M_PARTS = [
    "cc3m_96px_part1",
    "cc3m_96px_part2",
    "cc3m_96px_part3",
    "cc3m_96px_part4",
    "cc3m_96px_part5"
]

# --------------------------
# LARS Optimizer
# --------------------------
class LARS(torch.optim.Optimizer):
    """
    LARS optimizer (Layer-wise Adaptive Rate Scaling)
    Used in SimCLR, BYOL, VicReg, etc.
    """
    def __init__(self, params, lr=0, weight_decay=0, momentum=0.9, eta=0.001,
                 weight_decay_filter=None, lars_adaptation_filter=None):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum,
                       eta=eta, weight_decay_filter=weight_decay_filter,
                       lars_adaptation_filter=lars_adaptation_filter)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            for p in g['params']:
                dp = p.grad

                if dp is None:
                    continue

                if g['weight_decay_filter'] is None or not g['weight_decay_filter'](p):
                    dp = dp.add(p, alpha=g['weight_decay'])

                if g['lars_adaptation_filter'] is None or not g['lars_adaptation_filter'](p):
                    param_norm = torch.norm(p)
                    update_norm = torch.norm(dp)
                    one = torch.ones_like(param_norm)
                    q = torch.where(param_norm > 0.,
                                  torch.where(update_norm > 0,
                                            (g['eta'] * param_norm / update_norm), one),
                                  one)
                    dp = dp.mul(q)

                param_state = self.state[p]
                if 'mu' not in param_state:
                    param_state['mu'] = torch.zeros_like(p)
                mu = param_state['mu']
                mu.mul_(g['momentum']).add_(dp)

                p.add_(mu, alpha=-g['lr'])

def exclude_bias_and_norm(p):
    """Exclude bias and norm layers from LARS adaptation"""
    return p.ndim == 1

# --------------------------
# Distributed Training Setup
# --------------------------
def setup_distributed():
    """Initialize distributed training"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        gpu = int(os.environ['LOCAL_RANK'])
    else:
        rank = 0
        world_size = 1
        gpu = 0
    
    if world_size > 1:
        dist.init_process_group(
            backend='nccl',
            init_method='env://',
            world_size=world_size,
            rank=rank
        )
        torch.cuda.set_device(gpu)
    
    return rank, world_size, gpu

def cleanup_distributed():
    """Clean up distributed training"""
    if dist.is_initialized():
        dist.destroy_process_group()

# --------------------------
# Helpers
# --------------------------
def set_seed(s: int):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)

def save_ckpt(state: dict, path: Path):
    tmp = path.with_suffix(".tmp")
    torch.save(state, tmp)
    tmp.replace(path)

def load_ckpt(path: Path, map_location="cpu"):
    if not path or not Path(path).exists():
        return None
    return torch.load(path, map_location=map_location)

def make_cosine_scheduler(optimizer, total_steps, warmup_steps=0):
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

# --------------------------
# Data Augmentation
# --------------------------
NORM = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

def get_vicreg_augmentation(size=96):
    """VicReg style augmentation"""
    aug = transforms.Compose([
        transforms.RandomResizedCrop(size, scale=(0.08, 1.0)),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=7, sigma=(0.1, 2.0))], p=0.5),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        NORM
    ])
    return aug

class MultiFolderImageDataset(Dataset):
    """Collect images from multiple folders"""
    def __init__(self, roots: List[str], exts=None):
        exts = exts or [".jpg", ".jpeg", ".png", ".JPEG", ".PNG"]
        files = []
        for r in roots:
            p = Path(r)
            if not p.exists():
                continue
            for e in exts:
                files += list(p.rglob(f"*{e}"))
        self.files = sorted([f for f in files if f.is_file()])
        if len(self.files) == 0:
            raise RuntimeError(f"No images found in {roots}")
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        try:
            with Image.open(self.files[idx]) as im:
                return im.convert("RGB")
        except:
            idx = random.randint(0, len(self.files) - 1)
            with Image.open(self.files[idx]) as im:
                return im.convert("RGB")

class VicRegDataset(Dataset):
    def __init__(self, roots: List[str], transform):
        self.ds = MultiFolderImageDataset(roots)
        self.transform = transform
    
    def __len__(self):
        return len(self.ds)
    
    def __getitem__(self, idx):
        img = self.ds[idx]
        return self.transform(img), self.transform(img)

# --------------------------
# Model: VicReg with ConvNeXt
# --------------------------
def get_convnext_base(device="cuda"):
    """Get ConvNeXt-Base backbone"""
    try:
        # ConvNeXt-Base: ~88M params
        from torchvision.models import convnext_base
        model = convnext_base(weights=None)
        # Remove classifier
        feat_dim = model.classifier[2].in_features
        model.classifier = nn.Identity()
    except:
        raise RuntimeError("ConvNeXt-Base not available")
    
    model.to(device)
    return model, feat_dim

class Expander(nn.Module):
    """VicReg expander (3-layer MLP)"""
    def __init__(self, in_dim, hidden_dim=8192, out_dim=8192):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim)
        )
    
    def forward(self, x):
        return self.net(x)

class VicReg(nn.Module):
    """VicReg model"""
    def __init__(self, backbone="convnext_base", proj_dim=8192, device="cuda"):
        super().__init__()
        self.backbone, feat_dim = get_convnext_base(device)
        self.expander = Expander(feat_dim, hidden_dim=proj_dim, out_dim=proj_dim)
    
    def forward(self, x1, x2):
        """
        x1, x2: two augmented views
        Returns: embeddings for both views
        """
        z1 = self.backbone(x1)
        z1 = self.expander(z1)
        
        z2 = self.backbone(x2)
        z2 = self.expander(z2)
        
        return z1, z2

# --------------------------
# VicReg Loss
# --------------------------
def off_diagonal(x):
    """Return off-diagonal elements of a square matrix"""
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

def vicreg_loss(z1, z2, sim_coeff=25.0, std_coeff=25.0, cov_coeff=1.0):
    """
    VicReg loss with three components:
    1. Invariance (similarity): MSE between representations
    2. Variance: Hinge loss to maintain std > 1
    3. Covariance: Decorrelate dimensions
    """
    batch_size = z1.shape[0]
    repr_dim = z1.shape[1]
    
    # Invariance loss (MSE)
    sim_loss = F.mse_loss(z1, z2)
    
    # Variance loss
    std_z1 = torch.sqrt(z1.var(dim=0) + 1e-4)
    std_z2 = torch.sqrt(z2.var(dim=0) + 1e-4)
    std_loss = torch.mean(F.relu(1 - std_z1)) + torch.mean(F.relu(1 - std_z2))
    
    # Covariance loss
    z1 = z1 - z1.mean(dim=0)
    z2 = z2 - z2.mean(dim=0)
    
    cov_z1 = (z1.T @ z1) / (batch_size - 1)
    cov_z2 = (z2.T @ z2) / (batch_size - 1)
    
    cov_loss = off_diagonal(cov_z1).pow_(2).sum() / repr_dim + \
               off_diagonal(cov_z2).pow_(2).sum() / repr_dim
    
    # Total loss
    loss = sim_coeff * sim_loss + std_coeff * std_loss + cov_coeff * cov_loss
    
    return loss, sim_loss, std_loss, cov_loss

# --------------------------
# Training
# --------------------------
def train_vicreg(pretrain_parts: List[str], cfg: dict, rank=0, world_size=1):
    """Train VicReg model"""
    device = torch.device(f"cuda:{rank}")
    is_main = (rank == 0)
    
    # Data
    transform = get_vicreg_augmentation(size=96)
    dataset = VicRegDataset(pretrain_parts, transform)
    
    if world_size > 1:
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
        loader = DataLoader(dataset, batch_size=cfg["batch_size"], sampler=sampler,
                          num_workers=cfg["num_workers"], pin_memory=True)
    else:
        loader = DataLoader(dataset, batch_size=cfg["batch_size"], shuffle=True,
                          num_workers=cfg["num_workers"], pin_memory=True)
    
    # Model
    model = VicReg(
        backbone=cfg["backbone"],
        proj_dim=cfg["proj_dim"],
        device=device
    ).to(device)
    
    # torch.compile
    if cfg.get("compile", False) and hasattr(torch, 'compile'):
        if is_main:
            print("Using torch.compile...")
        model = torch.compile(model, mode="reduce-overhead")
    
    # DDP
    if world_size > 1:
        model = DDP(model, device_ids=[rank])
    
    # LARS Optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = LARS(
        params,
        lr=cfg["base_lr"],
        weight_decay=cfg["weight_decay"],
        momentum=0.9,
        eta=0.001,
        weight_decay_filter=exclude_bias_and_norm,
        lars_adaptation_filter=exclude_bias_and_norm
    )
    
    # Scheduler
    total_steps = cfg["epochs"] * len(loader)
    warmup_steps = cfg["warmup_epochs"] * len(loader)
    scheduler = make_cosine_scheduler(optimizer, total_steps, warmup_steps)
    
    # Training loop
    global_step = 0
    best_loss = float("inf")
    losses_history = []
    
    # bfloat16 context
    use_bf16 = cfg.get("use_bfloat16", False) and torch.cuda.is_bf16_supported()
    if use_bf16 and is_main:
        print("Using bfloat16 precision")
    
    model.train()
    for epoch in range(1, cfg["epochs"] + 1):
        if world_size > 1:
            sampler.set_epoch(epoch)
        
        running_loss = 0.0
        if is_main:
            pbar = tqdm(loader, desc=f"Epoch {epoch}/{cfg['epochs']}")
        else:
            pbar = loader
        
        for x1, x2 in pbar:
            global_step += 1
            x1 = x1.to(device, non_blocking=True)
            x2 = x2.to(device, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)
            
            # Forward with bfloat16
            if use_bf16:
                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    z1, z2 = model(x1, x2)
                    loss, sim_loss, std_loss, cov_loss = vicreg_loss(
                        z1, z2, 
                        cfg["sim_coeff"], 
                        cfg["std_coeff"], 
                        cfg["cov_coeff"]
                    )
            else:
                z1, z2 = model(x1, x2)
                loss, sim_loss, std_loss, cov_loss = vicreg_loss(
                    z1, z2,
                    cfg["sim_coeff"],
                    cfg["std_coeff"],
                    cfg["cov_coeff"]
                )
            
            # Backward
            loss.backward()
            
            # Gradient clipping
            if cfg.get("grad_clip", 0) > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            
            optimizer.step()
            scheduler.step()
            
            running_loss += loss.item()
            
            if is_main and global_step % cfg["print_every"] == 0:
                pbar.set_postfix(
                    loss=f"{loss.item():.4f}",
                    sim=f"{sim_loss.item():.3f}",
                    std=f"{std_loss.item():.3f}",
                    cov=f"{cov_loss.item():.3f}",
                    lr=f"{optimizer.param_groups[0]['lr']:.3e}"
                )
        
        avg_loss = running_loss / len(loader)
        losses_history.append(avg_loss)
        
        if is_main:
            print(f"[Epoch {epoch}] avg_loss: {avg_loss:.4f}")
            
            # Save checkpoint
            Path(cfg["ckpt_dir"]).mkdir(exist_ok=True)
            state = {
                "epoch": epoch,
                "model": model.module.state_dict() if world_size > 1 else model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "loss": avg_loss
            }
            save_ckpt(state, Path(cfg["ckpt_dir"]) / f"model_ep{epoch}.pth")
            
            if avg_loss < best_loss:
                best_loss = avg_loss
                save_ckpt(state, Path(cfg["ckpt_dir"]) / "model_best.pth")
    
    # Save final
    if is_main:
        final_state = {
            "epoch": cfg["epochs"],
            "model": model.module.state_dict() if world_size > 1 else model.state_dict(),
            "loss": best_loss
        }
        save_ckpt(final_state, Path(cfg["ckpt_dir"]) / "model_final.pth")
        
        # Save losses
        np.save(Path(cfg["ckpt_dir"]) / "losses.npy", np.array(losses_history))
    
    return model

# --------------------------
# Main
# --------------------------
def main():
    # Setup
    rank, world_size, gpu = setup_distributed()
    set_seed(DEFAULTS["seed"] + rank)
    
    cfg = DEFAULTS.copy()
    cfg["ckpt_dir"] = "./ckpts_vicreg"
    
    # Train
    if rank == 0:
        print("="*70)
        print("VicReg Training - ConvNeXt-Base")
        print("="*70)
        print(f"Backbone: {cfg['backbone']}")
        print(f"Epochs: {cfg['epochs']}")
        print(f"Batch size: {cfg['batch_size']}")
        print(f"Learning rate: {cfg['base_lr']}")
        print(f"Weight decay: {cfg['weight_decay']}")
        print(f"Optimizer: LARS")
        print(f"Projection dim: {cfg['proj_dim']}")
        print(f"VicReg coeffs: sim={cfg['sim_coeff']}, std={cfg['std_coeff']}, cov={cfg['cov_coeff']}")
        print(f"torch.compile: {cfg.get('compile', False)}")
        print(f"bfloat16: {cfg.get('use_bfloat16', False)}")
        print(f"Gradient clipping: {cfg.get('grad_clip', 0)}")
        print(f"World size: {world_size}")
        print("="*70)
    
    model = train_vicreg(CC3M_PARTS, cfg, rank, world_size)
    
    cleanup_distributed()

if __name__ == "__main__":
    main()

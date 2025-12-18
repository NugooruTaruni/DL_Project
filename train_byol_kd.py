#!/usr/bin/env python3
"""
train_byol_kd.py

BYOL-based training with teacher-student setup:
- Teacher: Vision Transformer (ViT-B/16)
- Student: ResNet-50
- Pretraining datasets (use multiple for best results):
  * CC3M
  * COCO: Object detection dataset 
- Two-view augmentations, mixed precision.
- Choose optimizer via --optim (adamw | lars).

"""

import argparse, math, random, time, json, io
from pathlib import Path
from typing import List
import numpy as np
from PIL import Image
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from tqdm import tqdm

# --------------------------
# Config defaults
# --------------------------
DEFAULTS = {
    "ckpt_dir": "./ckpts_byol",
    "device": "cuda",
    "num_workers": 8,
    "batch_size": 256,
    "teacher_epochs": 200,
    "student_epochs": 250,
    "teacher_backbone": "vit_base",
    "student_backbone": "resnet50",
    "optim": "adamw",
    "base_lr_adamw": 1e-4,
    "base_lr_lars": 0.2,
    "weight_decay": 1e-4,
    "warmup_epochs": 10,
    "m_base": 0.996,
    "print_every": 200,
    "prefetch_factor": 4,
    "seed": 42,
    "max_fallback_attempts": 20,
}

# --------------------------
# Dataset Paths (Update these for your setup)
# --------------------------
"""
Pretraining Datasets Used:
1. CC3M (Conceptual Captions) - ~500K images
   - Already have: cc3m_96px_part1-5
   - Web images with diverse objects
   
2. COCO Unlabeled 2017 - 123,287 images (~19GB)
   - Download 


"""

# SET YOUR PATHS HERE
COCO_UNLABELED_PATH = "./coco_unlabeled/unlabeled2017"  
CC3M_PARTS = [
    "cc3m_96px_part1",
    "cc3m_96px_part2", 
    "cc3m_96px_part3",
    "cc3m_96px_part4",
    "cc3m_96px_part5"
]

# --------------------------
# Helpers
# --------------------------

def path_allowed(p: Path):
    low = str(p).lower()
    return not any(x in low for x in BAD_SUBSTRS)

def set_seed(s: int):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def save_ckpt(state: dict, path: Path):
    tmp = path.with_suffix(".tmp")
    torch.save(state, tmp)
    tmp.replace(path)

def load_ckpt(path: Path, map_location="cpu"):
    if not path or not Path(path).exists(): return None
    return torch.load(path, map_location=map_location)

def make_cosine_scheduler(optimizer, total_steps, warmup_steps=0):
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

# --------------------------
# Dataset: robust (skip corrupts)
# --------------------------
NORM = transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])

def get_two_views(size=96, strength="medium"):
    if strength == "weak":
        scale=(0.5,1.0); color=(0.2,0.2,0.1,0.05); cp=0.4
    elif strength == "medium":
        scale=(0.2,1.0); color=(0.4,0.4,0.2,0.1); cp=0.8
    else:
        scale=(0.08,1.0); color=(0.6,0.6,0.4,0.2); cp=0.9

    aug1 = transforms.Compose([
        transforms.RandomResizedCrop(size, scale=scale),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([transforms.ColorJitter(*color)], p=cp),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=7, sigma=(0.1,2.0))], p=0.5),
        transforms.ToTensor(), NORM
    ])
    aug2 = transforms.Compose([
        transforms.RandomResizedCrop(size, scale=scale),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([transforms.ColorJitter(*color)], p=cp*0.7),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=7, sigma=(0.1,2.0))], p=0.2),
        transforms.ToTensor(), NORM
    ])
    return aug1, aug2

class MultiFolderImageDataset(Dataset):
    """Collect images recursively, skip unreadable files by trying alternatives."""
    def __init__(self, roots: List[str], exts=None, max_fallback_attempts: int = 20, scan_and_filter: bool = False):
        exts = exts or [".jpg",".jpeg",".png",".JPEG",".PNG"]
        files = []
        for r in roots:
            p = Path(r)
            if not p.exists(): continue
            for e in exts:
                files += list(p.rglob(f"*{e}"))
        files = sorted([p for p in files if p.is_file() and path_allowed(p)])
        if len(files) == 0:
            raise RuntimeError("No allowed images found in provided roots (after filtering).")
        self.files = files
        self.max_fallback_attempts = int(max_fallback_attempts)

        if scan_and_filter:
            good = []
            for p in tqdm(self.files, desc="scanning images", leave=False):
                if self._try_open(p) is not None:
                    good.append(p)
            self.files = good
            if len(self.files) == 0:
                raise RuntimeError("No readable images after scan.")

    def __len__(self): return len(self.files)

    def _try_open(self, p: Path):
        try:
            with Image.open(p) as im:
                im.load(); return im.convert("RGB")
        except Exception:
            return None

    def __getitem__(self, idx):
        if idx < 0: idx = idx % len(self.files)
        img = self._try_open(self.files[idx])
        if img is not None:
            return img
        tried = {idx}; attempts = 0; n = len(self.files)
        while attempts < self.max_fallback_attempts:
            r = random.randint(0, n-1)
            if r in tried:
                attempts += 1; continue
            tried.add(r); attempts += 1
            img = self._try_open(self.files[r])
            if img is not None: return img
        raise RuntimeError(f"Failed to open any image after {self.max_fallback_attempts} attempts. Consider using --scan to pre-filter.")

class PretrainTwoViewDataset(Dataset):
    def __init__(self, roots: List[str], aug1, aug2, max_fallback_attempts=20, scan_and_filter=False):
        self.ds = MultiFolderImageDataset(roots, max_fallback_attempts=max_fallback_attempts, scan_and_filter=scan_and_filter)
        self.aug1, self.aug2 = aug1, aug2
    def __len__(self): return len(self.ds)
    def __getitem__(self, idx):
        img = self.ds[idx]
        return self.aug1(img), self.aug2(img)

# --------------------------
# Model: backbone + projector + predictor (BYOL)
# --------------------------
def get_backbone(name="resnet50", device="cuda"):
    """Get backbone - supports ViT and ResNet"""
    if name == "vit_base" or name == "vit_b_16":
        try:
            weights = models.ViT_B_16_Weights.DEFAULT
            m = models.vit_b_16(weights=weights)
        except Exception:
            m = models.vit_b_16(weights=None)
        m.heads = nn.Identity()
        dim = 768
    elif name == "resnet18":
        try:
            weights = models.ResNet18_Weights.DEFAULT
            m = models.resnet18(weights=weights)
        except Exception:
            m = models.resnet18(weights=None)
        dim = 512
        m.fc = nn.Identity()
    elif name == "resnet50":
        try:
            weights = models.ResNet50_Weights.DEFAULT
            m = models.resnet50(weights=weights)
        except Exception:
            m = models.resnet50(weights=None)
        dim = 2048
        m.fc = nn.Identity()
    else:
        raise ValueError(f"unsupported backbone: {name}")
    m.to(device)
    return m, dim

class MLP(nn.Module):
    def __init__(self, ind, hid=4096, outd=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(ind, hid),
            nn.BatchNorm1d(hid),
            nn.ReLU(inplace=True),
            nn.Linear(hid, outd),
        )
    def forward(self, x): return self.net(x)

class BYOLModel(nn.Module):
    def __init__(self, backbone="resnet50", device="cuda"):
        super().__init__()
        self.device = device
        self.online_backbone, d = get_backbone(backbone, device=device)
        self.online_proj = MLP(d, hid=4096, outd=256)
        self.online_pred = MLP(256, hid=512, outd=256)
        self.target_backbone, _ = get_backbone(backbone, device=device)
        self.target_proj = MLP(d, hid=4096, outd=256)
        self._init_target()

    def _init_target(self):
        # copy online -> target
        for o, t in zip(self.online_backbone.parameters(), self.target_backbone.parameters()):
            t.data.copy_(o.data); t.requires_grad = False
        for o, t in zip(self.online_proj.parameters(), self.target_proj.parameters()):
            t.data.copy_(o.data); t.requires_grad = False

    @torch.no_grad()
    def update_target(self, m):
        for o, t in zip(self.online_backbone.parameters(), self.target_backbone.parameters()):
            t.data = t.data * m + o.data * (1.0 - m)
        for o, t in zip(self.online_proj.parameters(), self.target_proj.parameters()):
            t.data = t.data * m + o.data * (1.0 - m)

    def forward_online(self, x):
        h = self.online_backbone(x)
        z = self.online_proj(h)
        p = self.online_pred(z)
        return h, z, p

    @torch.no_grad()
    def forward_target(self, x):
        h = self.target_backbone(x)
        z = self.target_proj(h)
        return z

    def forward(self, x1, x2):
        _, z1, p1 = self.forward_online(x1)
        _, z2, p2 = self.forward_online(x2)
        with torch.no_grad():
            t1 = self.forward_target(x1)
            t2 = self.forward_target(x2)
        return p1, p2, t1, t2

    def get_features(self, x):
        return self.online_backbone(x)

# --------------------------
# LARS implementation (layerwise adaptive)
# --------------------------
class LARS(torch.optim.Optimizer):
    def __init__(self, params, lr=0.1, momentum=0.9, weight_decay=1e-6, eta=0.001, eps=1e-8):
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay, eta=eta, eps=eps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            g_lr = group.get('lr', 0.0)
            for p in group['params']:
                if p.grad is None: continue
                d_p = p.grad
                p_norm = torch.norm(p.data)
                g_norm = torch.norm(d_p)
                if group.get('lars_exclude', False):
                    local_lr = 1.0
                else:
                    denom = (g_norm + group.get('weight_decay', 0.0) * p_norm + group.get('eps', 1e-8))
                    local_lr = group.get('eta', 0.001) * (p_norm / denom) if (p_norm > 0 and g_norm > 0) else 1.0
                if group.get('weight_decay', 0.0) != 0:
                    d_p = d_p.add(p.data, alpha=group['weight_decay'])
                if group.get('momentum', 0.0) != 0:
                    state = self.state[p]
                    if 'momentum_buffer' not in state:
                        buf = state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = state['momentum_buffer']
                        buf.mul_(group['momentum']).add_(d_p)
                    d_p = buf
                p.data.add_(d_p, alpha=-g_lr * local_lr)

def lars_param_groups(model, base_lr, weight_decay, lars_eta):
    decay = []; no_decay = []
    for name, p in model.named_parameters():
        if not p.requires_grad: continue
        if p.ndim == 1 or name.endswith(".bias") or "bn" in name.lower() or "norm" in name.lower():
            no_decay.append(p)
        else:
            decay.append(p)
    return [
        {"params": decay, "lr": base_lr, "weight_decay": weight_decay, "momentum": 0.9, "eta": lars_eta, "lars_exclude": False},
        {"params": no_decay, "lr": base_lr, "weight_decay": 0.0, "momentum": 0.9, "eta": lars_eta, "lars_exclude": True}
    ]

# --------------------------
# Losses & evaluation helpers
# --------------------------
def byol_loss_fn(p, t):
    p = F.normalize(p, dim=1); t = F.normalize(t, dim=1)
    return 2 - 2 * (p * t).sum(dim=1).mean()

def combined_byol_loss(p1, p2, t1, t2):
    return byol_loss_fn(p1, t2) + byol_loss_fn(p2, t1)

@torch.no_grad()
def extract_features_torch(model, loader, device):
    model.eval()
    feats = []; labs = []
    for batch in tqdm(loader, desc="Extract"):
        x, y = batch if isinstance(batch, tuple) and len(batch)==2 else (batch, None)
        x = x.to(device)
        f = model.get_features(x).cpu()
        feats.append(f)
        if y is not None:
            labs.append(y)
    feats = torch.cat(feats, dim=0)
    labs = torch.cat(labs, dim=0) if labs else None
    return feats, labs

# --------------------------
# Training loops
# --------------------------
def build_train_loader(pretrain_parts, batch_size, num_workers, prefetch_factor, scan_and_filter, max_fallback_attempts):
    aug1, aug2 = get_two_views()
    ds = PretrainTwoViewDataset(pretrain_parts, aug1, aug2, max_fallback_attempts=max_fallback_attempts, scan_and_filter=scan_and_filter)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                        pin_memory=True, drop_last=True, persistent_workers=True, prefetch_factor=prefetch_factor)
    return loader, len(ds)

def train_teacher(pretrain_parts: List[str], cfg: dict, resume_ckpt: Path=None):
    device = cfg["device"]
    loader, dataset_size = build_train_loader(pretrain_parts, cfg["batch_size"], cfg["num_workers"],
                                              cfg["prefetch_factor"], cfg["scan"], cfg["max_fallback_attempts"])
    steps_per_epoch = len(loader)
    total_steps = cfg["teacher_epochs"] * steps_per_epoch
    model = BYOLModel(backbone=cfg["teacher_backbone"], device=device).to(device)

    # optimizer selection
    if cfg["optim"] == "lars":
        pg = lars_param_groups(model, cfg["base_lr"], cfg["weight_decay"], cfg["lars_eta"])
        opt = LARS(pg, lr=cfg["base_lr"], momentum=0.9, weight_decay=cfg["weight_decay"], eta=cfg["lars_eta"])
    else:
        # AdamW with no weight decay on biases/norms
        decay = []; no_decay = []
        for name, p in model.named_parameters():
            if not p.requires_grad: continue
            if p.ndim == 1 or name.endswith(".bias") or "bn" in name.lower() or "norm" in name.lower():
                no_decay.append(p)
            else:
                decay.append(p)
        param_groups = [{"params": decay, "weight_decay": cfg["weight_decay"]}, {"params": no_decay, "weight_decay": 0.0}]
        opt = torch.optim.AdamW(param_groups, lr=cfg["base_lr"], weight_decay=cfg["weight_decay"])

    scheduler = make_cosine_scheduler(opt, total_steps, warmup_steps=cfg["warmup_epochs"] * steps_per_epoch)
    scaler = torch.cuda.amp.GradScaler()
    start_epoch = 1; best_loss = float("inf"); global_step = 0

    # resume if checkpoint provided
    if resume_ckpt and resume_ckpt.exists():
        ck = load_ckpt(resume_ckpt, map_location=device)
        if ck:
            if "model" in ck: model.load_state_dict(ck["model"], strict=False)
            if "opt" in ck:
                try: opt.load_state_dict(ck.get("opt", opt.state_dict()))
                except Exception: print("Warning: couldn't load optimizer state.")
            start_epoch = ck.get("epoch", 1) + 1
            print("Resumed teacher from", resume_ckpt, "start_epoch", start_epoch)

    model.train()
    for epoch in range(start_epoch, cfg["teacher_epochs"] + 1):
        running = 0.0
        pbar = tqdm(loader, desc=f"Teacher epoch {epoch}/{cfg['teacher_epochs']}")
        for x1, x2 in pbar:
            global_step += 1
            x1 = x1.to(device, non_blocking=True); x2 = x2.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast():
                p1, p2, t1, t2 = model(x1, x2)
                loss = combined_byol_loss(p1, p2, t1, t2)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
            scaler.step(opt); scaler.update()
            scheduler.step()
            # EMA update
            m_base = cfg.get("m_base", 0.996)
            m = 1 - (1 - m_base) * (1 + math.cos(math.pi * global_step / max(1, total_steps))) / 2
            model.update_target(m)

            running += float(loss.item())
            if (global_step % cfg["print_every"]) == 0:
                lr_report = opt.param_groups[0].get('lr', cfg["base_lr"])
                pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{lr_report:.3e}")

        avg = running / len(loader)
        print(f"[Teacher] Epoch {epoch} avg_loss: {avg:.4f}")

        ck = {"epoch": epoch, "model": model.state_dict(), "opt": opt.state_dict(), "loss": avg}
        save_ckpt(ck, Path(cfg["ckpt_dir"]) / f"teacher_ep{epoch}.pth")
        if avg < best_loss:
            best_loss = avg
            save_ckpt({"epoch": epoch, "model": model.state_dict(), "loss": best_loss}, Path(cfg["ckpt_dir"]) / "teacher_best.pth")
            print("Saved teacher_best.pth")

    save_ckpt({"epoch": cfg["teacher_epochs"], "model": model.state_dict()}, Path(cfg["ckpt_dir"]) / "teacher_final.pth")
    return model

def train_student(pretrain_parts: List[str], teacher_ckpt: Path, cfg: dict, resume_ckpt: Path=None):
    """Train student with knowledge distillation from teacher"""
    device = cfg["device"]
    loader, dataset_size = build_train_loader(pretrain_parts, cfg["batch_size"], cfg["num_workers"],
                                              cfg["prefetch_factor"], cfg["scan"], cfg["max_fallback_attempts"])
    steps_per_epoch = len(loader)
    total_steps = cfg["student_epochs"] * steps_per_epoch
    
    # Load teacher
    teacher = BYOLModel(backbone=cfg["teacher_backbone"], device=device).to(device)
    if teacher_ckpt and teacher_ckpt.exists():
        ck = load_ckpt(teacher_ckpt, map_location=device)
        if ck and "model" in ck:
            teacher.load_state_dict(ck["model"], strict=False)
            print(f"Loaded teacher from {teacher_ckpt}")
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    
    # Initialize student
    student = BYOLModel(backbone=cfg["student_backbone"], device=device).to(device)

    # optimizer
    if cfg["optim"] == "lars":
        pg = lars_param_groups(student, cfg["base_lr"], cfg["weight_decay"], cfg["lars_eta"])
        opt = LARS(pg, lr=cfg["base_lr"], momentum=0.9, weight_decay=cfg["weight_decay"], eta=cfg["lars_eta"])
    else:
        decay = []; no_decay = []
        for name, p in student.named_parameters():
            if not p.requires_grad: continue
            if p.ndim == 1 or name.endswith(".bias") or "bn" in name.lower() or "norm" in name.lower():
                no_decay.append(p)
            else:
                decay.append(p)
        param_groups = [{"params": decay, "weight_decay": cfg["weight_decay"]}, {"params": no_decay, "weight_decay": 0.0}]
        opt = torch.optim.AdamW(param_groups, lr=cfg["base_lr"], weight_decay=cfg["weight_decay"])

    scheduler = make_cosine_scheduler(opt, total_steps, warmup_steps=cfg["warmup_epochs"] * steps_per_epoch)
    scaler = torch.cuda.amp.GradScaler()
    start_epoch = 1; best_loss = float("inf"); global_step = 0

    # resume if checkpoint provided
    if resume_ckpt and resume_ckpt.exists():
        ck = load_ckpt(resume_ckpt, map_location=device)
        if ck:
            if "model" in ck: student.load_state_dict(ck["model"], strict=False)
            if "opt" in ck:
                try: opt.load_state_dict(ck.get("opt", opt.state_dict()))
                except Exception: print("Warning: couldn't load optimizer state.")
            start_epoch = ck.get("epoch", 1) + 1
            print("Resumed student from", resume_ckpt, "start_epoch", start_epoch)

    student.train()
    for epoch in range(start_epoch, cfg["student_epochs"] + 1):
        running = 0.0
        pbar = tqdm(loader, desc=f"Student epoch {epoch}/{cfg['student_epochs']}")
        for x1, x2 in pbar:
            global_step += 1
            x1 = x1.to(device, non_blocking=True); x2 = x2.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            
            with torch.cuda.amp.autocast():
                # Student predictions
                sp1, sp2, st1, st2 = student(x1, x2)
                # Teacher predictions (detached)
                with torch.no_grad():
                    tp1, tp2, tt1, tt2 = teacher(x1, x2)
                
                # BYOL loss for student
                loss_byol = combined_byol_loss(sp1, sp2, st1, st2)
                # Knowledge distillation: student online matches teacher online
                loss_kd = byol_loss_fn(sp1, tp1) + byol_loss_fn(sp2, tp2)
                loss = loss_byol + loss_kd
                
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(student.parameters(), 3.0)
            scaler.step(opt); scaler.update()
            scheduler.step()
            
            # EMA update for student
            m_base = cfg.get("m_base", 0.996)
            m = 1 - (1 - m_base) * (1 + math.cos(math.pi * global_step / max(1, total_steps))) / 2
            student.update_target(m)

            running += float(loss.item())
            if (global_step % cfg["print_every"]) == 0:
                lr_report = opt.param_groups[0].get('lr', cfg["base_lr"])
                pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{lr_report:.3e}")

        avg = running / len(loader)
        print(f"[Student] Epoch {epoch} avg_loss: {avg:.4f}")

        ck = {"epoch": epoch, "model": student.state_dict(), "opt": opt.state_dict(), "loss": avg}
        save_ckpt(ck, Path(cfg["ckpt_dir"]) / f"student_ep{epoch}.pth")
        if avg < best_loss:
            best_loss = avg
            save_ckpt({"epoch": epoch, "model": student.state_dict(), "loss": best_loss}, Path(cfg["ckpt_dir"]) / "student_best.pth")
            print("Saved student_best.pth")

    save_ckpt({"epoch": cfg["student_epochs"], "model": student.state_dict()}, Path(cfg["ckpt_dir"]) / "student_final.pth")
    return student

# --------------------------
# Linear Probing for Evaluation
# --------------------------
class LinearProbeDataset(Dataset):
    """Dataset for linear probing with labels"""
    def __init__(self, root: str, transform=None):
        """
        Expects directory structure:
        root/
          class1/
            img1.jpg
            img2.jpg
          class2/
            img3.jpg
        """
        self.root = Path(root)
        self.transform = transform or transforms.Compose([
            transforms.Resize(96),
            transforms.CenterCrop(96),
            transforms.ToTensor(),
            NORM
        ])
        
        # Find all classes and images
        self.samples = []
        self.classes = sorted([d.name for d in self.root.iterdir() if d.is_dir()])
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        for class_name in self.classes:
            class_dir = self.root / class_name
            for ext in ['.jpg', '.jpeg', '.png', '.JPEG', '.PNG']:
                for img_path in class_dir.glob(f'*{ext}'):
                    self.samples.append((img_path, self.class_to_idx[class_name]))
        
        print(f"Found {len(self.samples)} images in {len(self.classes)} classes")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        try:
            img = Image.open(img_path).convert('RGB')
            img = self.transform(img)
            return img, label
        except Exception as e:
            # Return a random valid sample if this one fails
            return self.__getitem__(random.randint(0, len(self) - 1))

def linear_probe(model_ckpt: Path, train_dir: str, val_dir: str, cfg: dict, backbone_type="resnet50"):
    """
    Linear probing: Freeze backbone, train linear classifier
    """
    device = cfg["device"]
    
    # Load pretrained model
    print(f"\nLoading pretrained model from {model_ckpt}...")
    ckpt = load_ckpt(model_ckpt, map_location=device)
    if not ckpt or 'model' not in ckpt:
        raise ValueError(f"Invalid checkpoint: {model_ckpt}")
    
    # Initialize model and load weights
    backbone, feat_dim = get_backbone(backbone_type, device=device)
    
    # Load only backbone weights
    backbone_state = {}
    for k, v in ckpt['model'].items():
        if k.startswith('online_backbone.'):
            new_key = k.replace('online_backbone.', '')
            backbone_state[new_key] = v
    
    backbone.load_state_dict(backbone_state, strict=True)
    backbone.eval()
    
    # Freeze backbone
    for param in backbone.parameters():
        param.requires_grad = False
    
    print(f"✓ Loaded pretrained {backbone_type} backbone (frozen)")
    
    # Create datasets
    train_dataset = LinearProbeDataset(train_dir)
    val_dataset = LinearProbeDataset(val_dir)
    num_classes = len(train_dataset.classes)
    
    train_loader = DataLoader(train_dataset, batch_size=cfg["batch_size"], 
                            shuffle=True, num_workers=cfg["num_workers"], pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg["batch_size"], 
                          shuffle=False, num_workers=cfg["num_workers"], pin_memory=True)
    
    # Create linear classifier
    classifier = nn.Linear(feat_dim, num_classes).to(device)
    
    # Optimizer
    optimizer = torch.optim.Adam(classifier.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    best_acc = 0.0
    epochs = 100
    
    print(f"\n{'='*70}")
    print(f"LINEAR PROBING: {num_classes} classes, {len(train_dataset)} train, {len(val_dataset)} val")
    print(f"{'='*70}\n")
    
    for epoch in range(1, epochs + 1):
        # Train
        classifier.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)
            
            # Extract features (frozen backbone)
            with torch.no_grad():
                features = backbone(images)
            
            # Train classifier
            optimizer.zero_grad()
            outputs = classifier(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{100.*train_correct/train_total:.2f}%")
        
        scheduler.step()
        train_acc = 100. * train_correct / train_total
        
        # Validation
        classifier.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc="Validating", leave=False):
                images = images.to(device)
                labels = labels.to(device)
                
                features = backbone(images)
                outputs = classifier(features)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        val_acc = 100. * val_correct / val_total
        
        print(f"Epoch {epoch}: Train Acc={train_acc:.2f}% | Val Acc={val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                'epoch': epoch,
                'classifier': classifier.state_dict(),
                'backbone_type': backbone_type,
                'num_classes': num_classes,
                'accuracy': val_acc
            }, Path(cfg["ckpt_dir"]) / "linear_probe_best.pth")
            print(f"  ✓ New best: {val_acc:.2f}%")
    
    print(f"\n{'='*70}")
    print(f"LINEAR PROBING COMPLETE")
    print(f"Best Validation Accuracy: {best_acc:.2f}%")
    print(f"{'='*70}\n")
    
    return best_acc

# --------------------------
# CLI
# --------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["teacher","student","extract"], required=True, 
                   help="teacher = train teacher; student = train student with KD; extract = save features")
    p.add_argument("--pretrain", nargs="+", required=True, help="one or more pretrain folder paths (unlabeled)")
    p.add_argument("--teacher_ckpt", default=None, help="teacher checkpoint for student training")
    p.add_argument("--ckpt_dir", default=DEFAULTS["ckpt_dir"])
    p.add_argument("--resume", default=None, help="resume checkpoint path")
    p.add_argument("--device", default=DEFAULTS["device"])
    p.add_argument("--num_workers", type=int, default=DEFAULTS["num_workers"])
    p.add_argument("--batch_size", type=int, default=DEFAULTS["batch_size"])
    p.add_argument("--teacher_epochs", type=int, default=DEFAULTS["teacher_epochs"])
    p.add_argument("--student_epochs", type=int, default=DEFAULTS["student_epochs"])
    p.add_argument("--teacher_backbone", default=DEFAULTS["teacher_backbone"])
    p.add_argument("--student_backbone", default=DEFAULTS["student_backbone"])
    p.add_argument("--optim", choices=["adamw","lars"], default=DEFAULTS["optim"])
    p.add_argument("--base_lr", type=float, default=None, help="override base lr (auto default if omitted)")
    p.add_argument("--weight_decay", type=float, default=DEFAULTS["weight_decay"])
    p.add_argument("--lars_eta", type=float, default=0.001)
    p.add_argument("--warmup_epochs", type=int, default=DEFAULTS["warmup_epochs"])
    p.add_argument("--m_base", type=float, default=DEFAULTS["m_base"])
    p.add_argument("--print_every", type=int, default=DEFAULTS["print_every"])
    p.add_argument("--prefetch_factor", type=int, default=DEFAULTS["prefetch_factor"])
    p.add_argument("--scan", action="store_true", help="pre-scan and filter unreadable files (slow)")
    p.add_argument("--max_fallback_attempts", type=int, default=DEFAULTS["max_fallback_attempts"])
    return p.parse_args()

def main():
    args = parse_args()
    cfg = DEFAULTS.copy()
    cfg["ckpt_dir"] = args.ckpt_dir
    cfg["device"] = torch.device(args.device if torch.cuda.is_available() else "cpu")
    cfg["num_workers"] = args.num_workers
    cfg["batch_size"] = args.batch_size
    cfg["teacher_epochs"] = args.teacher_epochs
    cfg["student_epochs"] = args.student_epochs
    cfg["teacher_backbone"] = args.teacher_backbone
    cfg["student_backbone"] = args.student_backbone
    cfg["optim"] = args.optim
    cfg["weight_decay"] = args.weight_decay
    cfg["lars_eta"] = args.lars_eta
    cfg["warmup_epochs"] = args.warmup_epochs
    cfg["m_base"] = args.m_base
    cfg["print_every"] = args.print_every
    cfg["prefetch_factor"] = args.prefetch_factor
    cfg["scan"] = bool(args.scan)
    cfg["max_fallback_attempts"] = int(args.max_fallback_attempts)

    # set sensible base_lr if user did not provide
    if args.base_lr is not None:
        cfg["base_lr"] = args.base_lr
    else:
        cfg["base_lr"] = DEFAULTS["base_lr_adamw"] if cfg["optim"] == "adamw" else DEFAULTS["base_lr_lars"]

    Path(cfg["ckpt_dir"]).mkdir(parents=True, exist_ok=True)
    set_seed(DEFAULTS["seed"])
    print("CONFIG:", json.dumps({k:v for k,v in cfg.items() if k not in ("device","pretrain_parts")}, default=str, indent=2))

    if args.mode == "teacher":
        if not args.pretrain:
            raise ValueError("--pretrain required for teacher training")
        resume_ckpt = Path(args.resume) if args.resume else None
        model = train_teacher(args.pretrain, cfg, resume_ckpt)
        print("Teacher finished. Best checkpoint at", Path(cfg["ckpt_dir"]) / "teacher_best.pth")
    elif args.mode == "student":
        if not args.pretrain:
            raise ValueError("--pretrain required for student training")
        if not args.teacher_ckpt:
            raise ValueError("--teacher_ckpt required for student training")
        teacher_ckpt = Path(args.teacher_ckpt)
        resume_ckpt = Path(args.resume) if args.resume else None
        model = train_student(args.pretrain, teacher_ckpt, cfg, resume_ckpt)
        print("Student finished. Best checkpoint at", Path(cfg["ckpt_dir"]) / "student_best.pth")
    elif args.mode == "linear_probe":
        if not args.model_ckpt:
            raise ValueError("--model_ckpt required for linear probing")
        if not args.train_dir or not args.val_dir:
            raise ValueError("--train_dir and --val_dir required for linear probing")
        model_ckpt = Path(args.model_ckpt)
        accuracy = linear_probe(model_ckpt, args.train_dir, args.val_dir, cfg, args.backbone_type)
        print(f"Linear probing complete. Best accuracy: {accuracy:.2f}%")
    else:
        # quick feature dump using pretrained backbone for downstream work (no training)
        if not args.pretrain:
            raise ValueError("--pretrain required for feature extraction")
        print("Mode 'extract' loads pretrained backbone and writes features for provided folders.")
        device = cfg["device"]
        backbone_name = args.teacher_backbone if hasattr(args, 'teacher_backbone') else "resnet50"
        backbone, _ = get_backbone(backbone_name, device=device)
        tf = transforms.Compose([transforms.Resize(96), transforms.CenterCrop(96), transforms.ToTensor(), NORM])
        ds = MultiFolderImageDataset(args.pretrain, max_fallback_attempts=cfg["max_fallback_attempts"], scan_and_filter=cfg["scan"])
        loader = DataLoader(ds, batch_size=cfg["batch_size"], shuffle=False, num_workers=cfg["num_workers"], pin_memory=True)
        backbone.eval()
        feats = []
        for img in tqdm(loader, desc="extract"):
            # loader returns PIL images; apply tf inside batch
            batch_t = torch.stack([tf(im) for im in img]).to(device)
            with torch.no_grad():
                f = backbone(batch_t).cpu()
            feats.append(f)
        feats = torch.cat(feats, dim=0)
        outp = Path(cfg["ckpt_dir"]) / "unlabeled_feats.pt"
        torch.save(feats, outp)
        print("Saved features to", outp)

if __name__ == "__main__":
    main()

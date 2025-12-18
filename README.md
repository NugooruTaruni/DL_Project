# Self-Supervised Visual Representation Learning

Self-supervised learning method for learning visual features from an unlabeled image corpus.

## Task

Develop a self-supervised learning (SSL) method to learn visual representations from ~500k unlabeled images. After pretraining, the encoder is frozen and evaluated using k-NN classification on a downstream task.

## Dataset

Data available at: [tsbpp/fall2025_deeplearning](https://huggingface.co/datasets/tsbpp/fall2025_deeplearning)

- `pretrain/` - Unlabeled pretraining images (~500k)
- `eval_public/train` - Labeled training split for evaluation
- `eval_public/test` - Labeled test split for evaluation

- Can use any extrernal datasets for pretraining.

All images are 96x96 pixels.

## Constraints

- Backbone must have < 100M parameters
- Model must be randomly initialized
- Encoder remains frozen during evaluation
- No training or adaptation on test images
- Image resolution fixed at 96px

## Evaluation

- Primary metric: Top-1 accuracy using k-NN on frozen features
- Linear probing may be reported but leaderboard uses k-NN
- Final ranking includes evaluation on a private held-out set


GDrive Link for the results and .pth files - https://drive.google.com/drive/folders/16upxf123FxnRTv91rrtnVB20iIZJuAaf?usp=sharing

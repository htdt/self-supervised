# Self-Supervised Representation Learning

This repository includes 3 types of losses:
- W-MSE
- Contrastive
- BYOL

And 4 datasets:
- CIFAR-10 and CIFAR-100
- STL-10
- Tiny ImageNet

Checkpoints are stored in `data` each 100 epochs during training.

## Installation

The implementation is based on PyTorch. Logging works on [wandb.ai](https://wandb.ai/). See `docker/Dockerfile`.

## Usage

Configuration has good default settings, to see all options:
```
python -m train --help
python -m test --help
```

To reproduce the results:
```
# W-MSE 4
python -m train --dataset cifar10 --epoch 1000 --lr 3e-3 --num_samples 4 --bs 256 --emb 64 --w_size 128
python -m train --dataset cifar100 --epoch 1000 --lr 3e-3 --num_samples 4 --bs 256 --emb 64 --w_size 128
python -m train --dataset stl10 --epoch 2000 --lr 2e-3 --num_samples 4 --bs 256 --emb 128 --w_size 256
python -m train --dataset tiny_in --epoch 1000 --lr 2e-3 --num_samples 4 --bs 256 --emb 128 --w_size 256

# W-MSE 2
python -m train --dataset cifar10 --epoch 1000 --lr 3e-3 --emb 64 --w_size 128
python -m train --dataset cifar100 --epoch 1000 --lr 3e-3 --emb 64 --w_size 128
python -m train --dataset stl10 --epoch 2000 --lr 2e-3 --emb 128 --w_size 256 --w_iter 4
python -m train --dataset tiny_in --epoch 1000 --lr 2e-3 --emb 128 --w_size 256 --w_iter 4

# Contrastive
python -m train --dataset cifar10 --epoch 1000 --lr 3e-3 --emb 64 --method contrastive
python -m train --dataset cifar100 --epoch 1000 --lr 3e-3 --emb 64 --method contrastive
python -m train --dataset stl10 --epoch 2000 --lr 2e-3 --emb 128 --method contrastive
python -m train --dataset tiny_in --epoch 1000 --lr 2e-3 --emb 128 --method contrastive

# BYOL
python -m train --dataset cifar10 --epoch 1000 --lr 3e-3 --emb 64 --method byol
python -m train --dataset cifar100 --epoch 1000 --lr 3e-3 --emb 64 --method byol
python -m train --dataset stl10 --epoch 2000 --lr 2e-3 --emb 128 --method byol
python -m train --dataset tiny_in --epoch 1000 --lr 2e-3 --emb 128 --method byol
```

# Self-Supervised Representation Learning

Official repository of **Whitening for Self-Supervised Representation Learning**.

It includes 3 types of losses:
- W-MSE [arxiv](https://arxiv.org/abs/2007.06346)
- Contrastive [SimCLR arxiv](https://arxiv.org/abs/2002.05709)
- BYOL [arxiv](https://arxiv.org/abs/2006.07733)

And 4 datasets:
- CIFAR-10 and CIFAR-100
- STL-10
- Tiny ImageNet

Checkpoints are stored in `data` each 100 epochs during training.

The implementation is optimized for a single GPU, although multiple are also supported. It includes fast evaluation: we pre-compute embeddings for the entire dataset and then train a classifier on top. The evaluation of the ResNet-18 encoder takes about one minute.

## Installation

The implementation is based on PyTorch. Logging works on [wandb.ai](https://wandb.ai/). See `docker/Dockerfile`.

## Usage

Detailed settings are good by default, to see all options:
```
python -m train --help
python -m test --help
```

To reproduce the results from [table 1](https://arxiv.org/abs/2007.06346):
#### W-MSE 4
```
python -m train --dataset cifar10 --epoch 1000 --lr 3e-3 --num_samples 4 --bs 256 --emb 64 --w_size 128
python -m train --dataset cifar100 --epoch 1000 --lr 3e-3 --num_samples 4 --bs 256 --emb 64 --w_size 128
python -m train --dataset stl10 --epoch 2000 --lr 2e-3 --num_samples 4 --bs 256 --emb 128 --w_size 256
python -m train --dataset tiny_in --epoch 1000 --lr 2e-3 --num_samples 4 --bs 256 --emb 128 --w_size 256
```

#### W-MSE 2
```
python -m train --dataset cifar10 --epoch 1000 --lr 3e-3 --emb 64 --w_size 128
python -m train --dataset cifar100 --epoch 1000 --lr 3e-3 --emb 64 --w_size 128
python -m train --dataset stl10 --epoch 2000 --lr 2e-3 --emb 128 --w_size 256 --w_iter 4
python -m train --dataset tiny_in --epoch 1000 --lr 2e-3 --emb 128 --w_size 256 --w_iter 4
```

#### Contrastive
```
python -m train --dataset cifar10 --epoch 1000 --lr 3e-3 --emb 64 --method contrastive
python -m train --dataset cifar100 --epoch 1000 --lr 3e-3 --emb 64 --method contrastive
python -m train --dataset stl10 --epoch 2000 --lr 2e-3 --emb 128 --method contrastive
python -m train --dataset tiny_in --epoch 1000 --lr 2e-3 --emb 128 --method contrastive
```

#### BYOL
```
python -m train --dataset cifar10 --epoch 1000 --lr 3e-3 --emb 64 --method byol
python -m train --dataset cifar100 --epoch 1000 --lr 3e-3 --emb 64 --method byol
python -m train --dataset stl10 --epoch 2000 --lr 2e-3 --emb 128 --method byol
python -m train --dataset tiny_in --epoch 1000 --lr 2e-3 --emb 128 --method byol
```

Use `--no_norm` to disable normalization (for Euclidean distance).

## Citation
```
@article{ermolov2020whitening,
  title={Whitening for Self-Supervised Representation Learning}, 
  author={Aleksandr Ermolov and Aliaksandr Siarohin and Enver Sangineto and Nicu Sebe},
  journal={arXiv preprint arXiv:2007.06346},
  year={2020}
}
```

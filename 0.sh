python -m train --emb 32 --dataset cifar10 --nce --no_norm --tau 1 --lr 1e-3 --l2 0
python -m train --emb 32 --dataset cifar10 --nce --lr 1e-3 --l2 0
python -m train --emb 32 --dataset cifar10 --mse --lr 1e-3 --l2 0
python -m train --emb 32 --dataset cifar10 --nce --mse --lr 1e-3 --l2 0

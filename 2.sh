python -m train --bs 128 --dataset cifar100 --mse
python -m train --bs 128 --dataset cifar100 --nce
python -m train --bs 128 --dataset cifar100 --nce --mse
python -m train --bs 128 --dataset cifar100 --nce --no_norm --tau 1
python -m train --bs 128 --dataset cifar100 --mse --w_iter 4 --w_slice 4
python -m train --bs 128 --dataset cifar100 --mse --w_iter 4 --w_slice 4 --nce

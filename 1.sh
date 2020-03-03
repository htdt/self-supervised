python -m train --emb 64 --dataset stl10 --mse --lr 1e-3 --l2 0
python -m train --emb 64 --dataset stl10 --nce --mse --lr 1e-3 --l2 0
python -m train --emb 64 --dataset stl10 --nce --lr 1e-3 --l2 0
python -m train --emb 64 --dataset stl10 --nce --no_norm --tau 1 --lr 1e-3 --l2 0

from collections import defaultdict
from tqdm import trange, tqdm
import numpy as np
import wandb
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.nn.functional import mse_loss, cross_entropy, normalize
import torch.backends.cudnn as cudnn

from model import get_model, get_head
from cfg import get_cfg
from eval_sgd import eval_sgd
import datasets


if __name__ == '__main__':
    cfg = get_cfg()
    wrun = wandb.init(project="white_ss", config=cfg, resume=cfg.resume)

    ds = getattr(datasets, cfg.dataset)
    loader_train = ds.loader_train(cfg.bs)
    loader_clf, loader_test = ds.loader_clf(), ds.loader_test()
    model, out_size = get_model(cfg.arch, cfg.dataset)
    params = list(model.parameters())

    if cfg.nce:
        head_nce = get_head(out_size, cfg.emb, cfg.linear_head)
        target_nce = torch.arange(cfg.bs).cuda()
        params += list(head_nce.parameters())
    if cfg.mse:
        head_mse = get_head(out_size, cfg.emb, cfg.linear_head,
                            whitening=True, multi_gpu=(cfg.w_slice > 1))
        params += list(head_mse.parameters())

    optimizer = optim.Adam(params, lr=cfg.lr, weight_decay=cfg.l2)
    scheduler = MultiStepLR(optimizer, milestones=cfg.drop)

    fname = f'data/{wrun.id}.pt'
    cudnn.benchmark = True
    for ep in trange(cfg.epoch_start, cfg.epoch, position=0):
        loss_ep = defaultdict(list)
        for (x0, x1), _ in tqdm(loader_train, position=1):
            optimizer.zero_grad()
            h0 = model(x0.cuda(non_blocking=True))
            h1 = model(x1.cuda(non_blocking=True))
            loss = 0

            if cfg.nce:
                z0, z1 = head_nce(h0), head_nce(h1)
                if cfg.norm:
                    z0 = normalize(z0, p=2, dim=1)
                    z1 = normalize(z1, p=2, dim=1)
                logits = z0 @ z1.t() / cfg.tau
                loss_nce = cross_entropy(logits, target_nce)
                loss_ep['nce'].append(loss_nce.item())
                loss += loss_nce

            if cfg.mse:
                h = torch.cat([h0, h1])
                if cfg.w_iter == 1 and cfg.w_slice == 1:
                    z = head_mse(h)
                    loss_mse = mse_loss(z[:len(h0)], z[len(h0):])
                else:
                    loss_mse = 0
                    num_slice = cfg.w_slice // torch.cuda.device_count()
                    for _ in range(cfg.w_iter):
                        z = torch.empty(len(h), cfg.emb, device='cuda')
                        perm = torch.randperm(len(h)).view(num_slice, -1)
                        for idx in perm:
                            z[idx] = head_mse(h[idx])
                        loss_mse += mse_loss(z[:len(h0)], z[len(h0):])
                    loss_mse /= cfg.w_iter
                loss_ep['mse'].append(loss_mse.item())
                loss += loss_mse

            loss.backward()
            optimizer.step()
            loss_ep['sum'].append(loss.item())
        scheduler.step()

        torch.save({
            'model': model.state_dict(),
            'head_nce': head_nce.state_dict() if cfg.nce else None,
            'head_mse': head_mse.state_dict() if cfg.mse else None,
            'optimizer': optimizer.state_dict(),
        }, fname)

        if (ep + 1) % cfg.eval_every == 0:
            acc = eval_sgd(model, out_size, loader_clf, loader_test, 500)
            wandb.log({'acc': acc}, commit=False)
            model.train()

        loss_ep = {k: np.mean(loss_ep[k]) for k in loss_ep}
        wandb.log({'loss': loss_ep, 'ep': ep})
    wandb.save(fname)

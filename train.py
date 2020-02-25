from collections import defaultdict
from tqdm import trange, tqdm
import numpy as np
import wandb
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.nn.functional import mse_loss, cross_entropy, normalize
import torch.backends.cudnn as cudnn

from model import get_model, Whitening2d, get_head
from cfg import get_cfg
from eval_sgd import eval_sgd
import cifar10
import stl10
import imagenet
DS = {'cifar10': cifar10, 'stl10': stl10, 'imagenet': imagenet}


if __name__ == '__main__':
    cfg = get_cfg()
    wrun = wandb.init(project="white_ss", config=cfg)

    loader_train = DS[cfg.dataset].loader_train(cfg.bs)
    loader_clf = DS[cfg.dataset].loader_clf()
    loader_test = DS[cfg.dataset].loader_test()
    model, out_size = get_model(cfg.arch, cfg.dataset)
    params = list(model.parameters())

    if cfg.xent:
        head_xent = get_head(out_size, cfg.emb, cfg.linear_head)
        target_xent = torch.arange(cfg.bs).cuda()
        params += list(head_xent.parameters())
    if cfg.mse:
        head_mse = get_head(out_size, cfg.emb, cfg.linear_head)
        whitening = Whitening2d(cfg.emb).cuda().train()
        params += list(head_mse.parameters()) + list(whitening.parameters())

    optimizer = optim.Adam(params, lr=cfg.lr, weight_decay=cfg.l2)
    scheduler = MultiStepLR(optimizer, milestones=cfg.drop)

    fname = f'data/{wrun.id}.pt'
    len10 = len(loader_train) // 10
    cudnn.benchmark = True
    for ep in trange(cfg.epoch, position=0):
        loss_ep = defaultdict(list)
        for (x0, x1), _ in tqdm(loader_train, position=1):
            optimizer.zero_grad()
            h0 = model(x0.cuda(non_blocking=True))
            h1 = model(x1.cuda(non_blocking=True))
            loss = 0

            if cfg.xent:
                z0, z1 = head_xent(h0), head_xent(h1)
                z0, z1 = normalize(z0, p=2, dim=1), normalize(z1, p=2, dim=1)
                logits = z0 @ z1.t() / cfg.tau
                loss_xent = cross_entropy(logits, target_xent)
                loss_ep['xent'].append(loss_xent.item())
                loss += cfg.xent_k * loss_xent

            if cfg.mse:
                bs = len(h0)
                z0, z1 = head_mse(h0), head_mse(h1)
                z = whitening(torch.cat([z0, z1]))
                z0, z1 = z[:bs], z[bs:]
                loss_mse = mse_loss(z0, z1)
                loss_ep['mse'].append(loss_mse.item())
                loss += loss_mse

            loss.backward()
            optimizer.step()
            loss_ep['sum'].append(loss.item())
            # if len(loss_ep) % len10 == 0:
            #    wandb.log({'loss10': np.mean(loss_ep[-len10:])})
        scheduler.step()

        torch.save({
            'model': model.state_dict(),
            'head_xent': head_xent.state_dict() if cfg.xent else None,
            'head_mse': head_mse.state_dict() if cfg.mse else None,
            'whitening': whitening.state_dict() if cfg.mse else None,
            'optimizer': optimizer.state_dict(),
        }, fname)

        if (ep + 1) % cfg.eval_every == 0:
            acc = eval_sgd(model, out_size, loader_clf, loader_test, 500)
            wandb.log({'acc': acc}, commit=False)
            model.train()

        loss_ep = {k: np.mean(loss_ep[k]) for k in loss_ep}
        wandb.log({'loss': loss_ep, 'ep': ep})
    wandb.save(fname)

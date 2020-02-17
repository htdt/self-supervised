from os import path
from time import time
from tqdm import trange, tqdm
import numpy as np
import wandb
import torch
import torch.optim as optim
from torch.nn.functional import mse_loss, cross_entropy
import torch.backends.cudnn as cudnn

from model import get_model, Whitening2d
from cfg import get_cfg
from clf import eval_lbfgs
import cifar10
import stl10
import imagenet
DS = {'cifar10': cifar10, 'stl10': stl10, 'imagenet': imagenet}


if __name__ == '__main__':
    cfg = get_cfg()
    wandb.init(project="white_ss", config=cfg)

    loader_train = DS[cfg.dataset].loader_train(cfg.bs)
    if cfg.eval_every != 0:
        loader_clf = DS[cfg.dataset].loader_clf()
        loader_test = DS[cfg.dataset].loader_test()
    model, head = get_model(cfg.arch, cfg.emb, cfg.dataset)
    params = list(model.parameters()) + list(head.parameters())

    if cfg.whitening:
        whitening = Whitening2d(cfg.emb)
        whitening.cuda()
        whitening.train()
        params += list(whitening.parameters())
    else:
        target = torch.arange(cfg.bs).cuda()

    optimizer = optim.Adam(params, lr=cfg.lr)
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=cfg.drop)

    def save(fname):
        torch.save({
            'model': model.state_dict(),
            'head': head.state_dict(),
            'optimizer': optimizer.state_dict(),
            'whitening': whitening.state_dict() if cfg.whitening else None,
        }, fname)

    cudnn.benchmark = True
    for ep in trange(cfg.epoch, position=0):
        loss_ep = []
        for x, _ in tqdm(loader_train, position=1):
            optimizer.zero_grad()
            x0 = head(model(x[0].cuda(non_blocking=True)))
            x1 = head(model(x[1].cuda(non_blocking=True)))

            if cfg.whitening:
                bs = len(x0)
                x = whitening(torch.cat([x0, x1]))
                loss = mse_loss(x[:bs], x[bs:])
            else:
                logits = x0 @ x1.t()
                loss = cross_entropy(logits, target)

            loss.backward()
            optimizer.step()
            loss_ep.append(loss.item())

        wandb.log({'loss': np.mean(loss_ep), 'ep': ep})
        scheduler.step()

        if (ep + 1) % cfg.save_every == 0:
            fname = path.join('data', f'{int(time())}.pt')
            save(fname)
            wandb.save(fname)
        elif cfg.checkpoint:
            fname = path.join('data', 'checkpoint.pt')
            save(fname)

        if cfg.eval_every != 0 and (ep + 1) % cfg.eval_every == 0:
            eval_lbfgs(model, loader_clf, loader_test)
            model.train()

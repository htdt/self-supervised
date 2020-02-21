from tqdm import trange, tqdm
import numpy as np
import wandb
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR, MultiStepLR
from torch.nn.functional import mse_loss, cross_entropy
import torch.backends.cudnn as cudnn
from torch.nn.functional import normalize

from model import get_model, Whitening2d
from cfg import get_cfg
from eval_lbfgs import eval_lbfgs
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
    model, head = get_model(cfg.arch, cfg.emb, cfg.dataset, cfg.big_head)
    params = list(model.parameters()) + list(head.parameters())

    if cfg.white:
        whitening = Whitening2d(cfg.emb)
        whitening.cuda()
        whitening.train()
        params += list(whitening.parameters())
    if cfg.loss == 'xent':
        target = torch.arange(cfg.bs).cuda()

    optimizer = optim.Adam(params, lr=cfg.lr, weight_decay=cfg.l2)
    if cfg.lr_exp:
        gamma = (cfg.lr_end / cfg.lr) ** (1 / cfg.epoch)
        scheduler = ExponentialLR(optimizer, gamma=gamma)
    else:
        scheduler = MultiStepLR(optimizer, milestones=cfg.drop)

    fname = f'data/{cfg.dataset}_{cfg.arch}_{cfg.loss}.pt'
    cudnn.benchmark = True
    for ep in trange(cfg.epoch, position=0):
        loss_ep = []
        for x, _ in tqdm(loader_train, position=1):
            optimizer.zero_grad()
            x0 = head(model(x[0].cuda(non_blocking=True)))
            x1 = head(model(x[1].cuda(non_blocking=True)))

            if cfg.white:
                bs = len(x0)
                x = whitening(torch.cat([x0, x1]))
                x0, x1 = x[:bs], x[bs:]

            if cfg.norm:
                x0 = normalize(x0, p=2, dim=1)
                x1 = normalize(x1, p=2, dim=1)

            if cfg.loss == 'mse':
                loss = mse_loss(x0, x1)
            elif cfg.loss == 'xent':
                logits = x0 @ x1.t() / cfg.tau
                loss = cross_entropy(logits, target)

            loss.backward()
            optimizer.step()
            loss_ep.append(loss.item())
            if len(loss_ep) % 100 == 0:
                wandb.log({'loss100': np.mean(loss_ep[-100:])})

        wandb.log({'loss': np.mean(loss_ep), 'ep': ep})
        scheduler.step()

        if (ep + 1) % cfg.save_every == 0:
            torch.save({
                'model': model.state_dict(),
                'head': head.state_dict(),
                'optimizer': optimizer.state_dict(),
                'whitening': whitening.state_dict() if cfg.white else None,
            }, fname)

        if cfg.eval_every != 0 and (ep + 1) % cfg.eval_every == 0:
            acc = eval_lbfgs(model, loader_clf, loader_test)
            wandb.log({'acc': acc})
            model.train()

    wandb.save(fname)

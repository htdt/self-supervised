from tqdm import trange, tqdm
import numpy as np
import wandb
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.nn.functional import mse_loss, cross_entropy, normalize
import torch.backends.cudnn as cudnn

from model import get_model, Whitening2d
from cfg import get_cfg
from eval_sgd import eval_sgd
import cifar10
import stl10
import imagenet
DS = {'cifar10': cifar10, 'stl10': stl10, 'imagenet': imagenet}


if __name__ == '__main__':
    cfg = get_cfg()
    wandb.init(project="white_ss", config=cfg)

    loader_train = DS[cfg.dataset].loader_train(cfg.bs)
    loader_clf = DS[cfg.dataset].loader_clf()
    loader_test = DS[cfg.dataset].loader_test()
    model, head = get_model(cfg.arch, cfg.emb, cfg.dataset, cfg.linear_head)
    out_size = head.module[0].in_features
    params = list(model.parameters()) + list(head.parameters())

    if cfg.white:
        whitening = Whitening2d(cfg.emb)
        whitening.cuda()
        whitening.train()
        params += list(whitening.parameters())
    if cfg.loss == 'xent':
        target = torch.arange(cfg.bs).cuda()

    optimizer = optim.Adam(params, lr=cfg.lr, weight_decay=cfg.l2)
    scheduler = MultiStepLR(optimizer, milestones=cfg.drop)

    fname = f'data/{cfg.dataset}_{cfg.arch}_{cfg.loss}.pt'
    len10 = len(loader_train) // 10
    cudnn.benchmark = True
    for ep in trange(cfg.epoch, position=0):
        loss_ep = []
        for (x0, x1), _ in tqdm(loader_train, position=1):
            optimizer.zero_grad()
            x0 = head(model(x0.cuda(non_blocking=True)))
            x1 = head(model(x1.cuda(non_blocking=True)))

            if cfg.norm:
                x0 = normalize(x0, p=2, dim=1)
                x1 = normalize(x1, p=2, dim=1)

            if cfg.white:
                bs = len(x0)
                x = whitening(torch.cat([x0, x1]))
                x0, x1 = x[:bs], x[bs:]

            if cfg.loss == 'mse':
                loss = mse_loss(x0, x1)
            elif cfg.loss == 'xent':
                logits = x0 @ x1.t() / cfg.tau
                loss = cross_entropy(logits, target)

            loss.backward()
            optimizer.step()
            loss_ep.append(loss.item())
            if len(loss_ep) % len10 == 0:
                wandb.log({'loss10': np.mean(loss_ep[-len10:])})
        scheduler.step()

        torch.save({
            'model': model.state_dict(),
            'head': head.state_dict(),
            'optimizer': optimizer.state_dict(),
            'whitening': whitening.state_dict() if cfg.white else None,
        }, fname)

        eval_ep = 50 if ep < cfg.drop[0] else 5
        if (ep + 1) % eval_ep == 0:
            acc = eval_sgd(model, out_size, loader_clf, loader_test)
            wandb.log({'acc': acc}, commit=False)
            model.train()

        wandb.log({'loss': np.mean(loss_ep), 'ep': ep})
    wandb.save(fname)

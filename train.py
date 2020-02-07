from time import time
from tqdm import trange
import numpy as np
import wandb
import torch
import torch.optim as optim
from torch.nn.functional import mse_loss, cross_entropy
import torch.backends.cudnn as cudnn
from dataset import get_loader_train
from model import get_model
from cfg import get_cfg
from dataset import get_loader_clf, get_loader_test
from clf import eval_lbfgs


if __name__ == '__main__':
    cfg = get_cfg()
    wandb.init(project="white_ss", config=cfg)

    cfgd = cfg.__dict__
    aug0 = {k[4:]: cfgd[k] for k in cfgd.keys() if k.startswith('im0_')}
    aug1 = {k[4:]: cfgd[k] for k in cfgd.keys() if k.startswith('im1_')}
    loader_train = get_loader_train(cfg.bs, aug0, aug1)
    loader_clf, loader_test = get_loader_clf(), get_loader_test()
    model, _ = get_model(cfg.arch, cfg.emb, cfg.whitening)
    model.train()

    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)
    if cfg.drop is not None:
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=cfg.drop)

    if not cfg.whitening:
        target = torch.arange(cfg.bs).cuda()

    def get_loss(x0, x1):
        if cfg.whitening:
            return mse_loss(x0, x1)
        else:
            logits = (x0 @ x1.t()) / cfg.nce_t
            return cross_entropy(logits, target)

    cudnn.benchmark = True
    for ep in trange(cfg.epoch):
        loss_ep = []
        for x, _ in loader_train:
            optimizer.zero_grad()
            x = [model(el.cuda(non_blocking=True)) for el in x]
            loss = get_loss(*x)
            loss.backward()
            optimizer.step()
            loss_ep.append(loss.item())

        wandb.log({'loss': np.mean(loss_ep), 'ep': ep})
        if cfg.drop is not None:
            scheduler.step()

        if (ep + 1) % cfg.save_every == 0:
            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
            fname = f'data/{cfg.arch}_{ep}_{int(time())}.pt'
            torch.save(checkpoint, fname)
            wandb.save(fname)

        if (ep + 1) % cfg.eval_every == 0:
            eval_lbfgs(model, loader_clf, loader_test)
            model.train()

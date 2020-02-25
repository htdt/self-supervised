from copy import deepcopy
from collections import defaultdict, deque
from tqdm import trange, tqdm
import numpy as np
import wandb
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.nn.functional import cross_entropy, normalize
import torch.backends.cudnn as cudnn

from model import get_model, get_head, lerp_nn
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

    head_xent = get_head(out_size, cfg.emb, cfg.linear_head)
    target_xent = torch.arange(cfg.bs).cuda()
    params += list(head_xent.parameters())

    model_t = deepcopy(model.module).eval()
    head_t = deepcopy(head_xent).eval()
    xq = deque(maxlen=(1024 * 16) // cfg.bs)

    optimizer = optim.Adam(params, lr=cfg.lr, weight_decay=cfg.l2)
    scheduler = MultiStepLR(optimizer, milestones=cfg.drop)

    fname = f'data/{wrun.id}.pt'
    len10 = len(loader_train) // 10
    cudnn.benchmark = True
    for ep in trange(cfg.epoch, position=0):
        loss_ep = defaultdict(list)
        for (x0, x1), _ in tqdm(loader_train, position=1):
            optimizer.zero_grad()
            z0 = head_xent(model(x0.cuda(non_blocking=True)))
            with torch.no_grad():
                z1 = head_t(model_t(x1.cuda(non_blocking=True)))

            z0, z1 = normalize(z0, p=2, dim=1), normalize(z1, p=2, dim=1)
            xq.appendleft(z1)
            z1 = torch.cat(list(xq))
            logits = z0 @ z1.t() / cfg.tau
            loss_xent = cross_entropy(logits, target_xent)
            loss_ep['xent'].append(loss_xent.item())

            loss_xent.backward()
            optimizer.step()
            lerp_nn(source=model.module, target=model_t, tau=.999)
        scheduler.step()

        torch.save({
            'model': model.state_dict(),
            'head_xent': head_xent.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, fname)

        if (ep + 1) % cfg.eval_every == 0:
            acc = eval_sgd(model, out_size, loader_clf, loader_test, 500)
            wandb.log({'acc': acc}, commit=False)
            model.train()

        loss_ep = {k: np.mean(loss_ep[k]) for k in loss_ep}
        wandb.log({'loss': loss_ep, 'ep': ep})
    wandb.save(fname)

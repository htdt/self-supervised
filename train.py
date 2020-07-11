from collections import defaultdict
from tqdm import trange, tqdm
import numpy as np
import wandb
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingWarmRestarts
from torch.nn.functional import mse_loss, cross_entropy, normalize
import torch.backends.cudnn as cudnn

from model import get_model, get_head
from cfg import get_cfg
from eval_sgd import eval_sgd
from datasets import get_ds


if __name__ == "__main__":
    cfg = get_cfg()
    wrun = wandb.init(project="white_ss", config=cfg)

    ds = get_ds(cfg.dataset)(cfg.bs, cfg)
    model, out_size = get_model(cfg.arch, cfg.dataset)
    params = list(model.parameters())

    if cfg.nce:
        head_nce = get_head(out_size, cfg.emb, cfg.head_layers)
        target_nce = torch.arange(cfg.bs).cuda()
        params += list(head_nce.parameters())
    if cfg.w_mse:
        head_mse = get_head(
            out_size,
            cfg.emb,
            layers=cfg.head_layers,
            whitening=True,
            w_eps=cfg.w_eps,
            method=cfg.method,
            add_bn=cfg.add_bn,
            add_bn_last=cfg.add_bn_last,
        )
        params += list(head_mse.parameters())

    optimizer = optim.Adam(
        params, lr=cfg.lr, betas=(0.8, 0.999), weight_decay=1e-5, eps=1e-8
    )
    if cfg.lr_step == "cos":
        scheduler = CosineAnnealingWarmRestarts(
            optimizer, T_0=cfg.T0, T_mult=cfg.Tmult, eta_min=cfg.eta_min
        )
    elif cfg.lr_step == "step":
        scheduler = MultiStepLR(optimizer, milestones=cfg.drop, gamma=cfg.drop_gamma)

    fname = f"data/{wrun.id}.pt"
    lr_warmup = 0 if cfg.lr_warmup else 500
    cudnn.benchmark = True
    for ep in trange(cfg.epoch, position=0):
        loss_ep = defaultdict(list)
        iters = len(ds.train)
        for n_iter, ((x0, x1), _) in enumerate(tqdm(ds.train, position=1)):
            if lr_warmup < 500:
                lr_scale = (lr_warmup + 1) / 500
                for pg in optimizer.param_groups:
                    pg["lr"] = cfg.lr * lr_scale
                lr_warmup += 1

            optimizer.zero_grad()
            h0 = model(x0.cuda(non_blocking=True))
            h1 = model(x1.cuda(non_blocking=True))
            loss = 0

            if cfg.nce:
                z0, z1 = head_nce(h0), head_nce(h1)
                if cfg.norm:
                    z0 = normalize(z0, p=2, dim=1)
                    z1 = normalize(z1, p=2, dim=1)
                logits = [z0 @ z1.t(), z0 @ z0.t(), z1 @ z1.t()]
                logits = torch.cat(logits, dim=1) / cfg.tau
                loss_nce = cross_entropy(logits, target_nce)
                loss_ep["nce"].append(loss_nce.item())
                loss += loss_nce

            if cfg.w_mse:
                h = torch.cat([h0, h1])
                if cfg.w_iter == 1 and cfg.w_slice == 1:
                    z = head_mse(h)
                    loss_mse = mse_loss(z[: len(h0)], z[len(h0) :])
                else:
                    loss_mse = 0
                    for _ in range(cfg.w_iter):
                        z = torch.empty(len(h), cfg.emb, device="cuda")
                        perm = torch.randperm(len(h)).view(cfg.w_slice, -1)
                        for idx in perm:
                            z[idx] = head_mse(h[idx])
                        loss_mse += mse_loss(z[: len(h0)], z[len(h0) :])
                    loss_mse /= cfg.w_iter
                loss_ep["mse"].append(loss_mse.item())
                loss += loss_mse

            loss.backward()
            optimizer.step()
            loss_ep["sum"].append(loss.item())
            if cfg.lr_step == "cos" and lr_warmup >= 500:
                scheduler.step(ep + n_iter / iters)

        if cfg.lr_step == "step":
            scheduler.step()

        if (ep + 1) % cfg.eval_every == 0:
            acc = eval_sgd(model, out_size, ds.clf, ds.test, 500)
            wandb.log({"acc": acc}, commit=False)
            model.train()

        loss_ep = {k: np.mean(loss_ep[k]) for k in loss_ep}
        wandb.log({"loss": loss_ep, "ep": ep})

    if cfg.save_model:
        torch.save(
            {
                "model": model.state_dict(),
                "head_nce": head_nce.state_dict() if cfg.nce else None,
                "head_mse": head_mse.state_dict() if cfg.w_mse else None,
                "optimizer": optimizer.state_dict(),
            },
            fname,
        )
        wandb.save(fname)

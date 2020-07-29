from tqdm import trange, tqdm
import numpy as np
import wandb
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingWarmRestarts
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

from model import get_model, get_head
from cfg import get_cfg
from eval_sgd import eval_sgd
from eval_knn import eval_knn
from datasets import get_ds
from whitening.cholesky import Whitening2d


def contrastive_loss(x0, x1, tau, norm=True):
    # https://github.com/google-research/simclr/blob/master/objective.py
    bsize = x0.shape[0]
    target = torch.arange(bsize).cuda()
    eye_mask = torch.eye(bsize).cuda() * 1e9
    if norm:
        x0 = F.normalize(x0, p=2, dim=1)
        x1 = F.normalize(x1, p=2, dim=1)
    logits00 = x0 @ x0.t() / tau - eye_mask
    logits11 = x1 @ x1.t() / tau - eye_mask
    logits01 = x0 @ x1.t() / tau
    logits10 = x1 @ x0.t() / tau
    return (
        F.cross_entropy(torch.cat([logits01, logits00], dim=1), target)
        + F.cross_entropy(torch.cat([logits10, logits11], dim=1), target)
    ) / 2


def norm_mse_loss(x0, x1):
    x0 = F.normalize(x0)
    x1 = F.normalize(x1)
    return 2 - 2 * (x0 * x1).sum(dim=-1).mean()


def w_mse_loss(x0, x1, whitening, w_iter, w_slice):
    bs = len(x0)
    x = torch.cat([x0, x1])
    loss = 0
    for _ in range(w_iter):
        z = torch.empty_like(x)
        perm = torch.randperm(len(x)).view(w_slice, -1)
        for idx in perm:
            z[idx] = whitening(x[idx])
        loss += norm_mse_loss(z[:bs], z[bs:])
    return loss / w_iter


if __name__ == "__main__":
    cfg = get_cfg()
    wrun = wandb.init(project="white_ss", config=cfg)

    ds = get_ds(cfg.dataset)(cfg.bs, cfg)
    model, out_size = get_model(cfg.arch, cfg.dataset)
    params = list(model.parameters())

    if cfg.w_mse:
        whitening = Whitening2d(cfg.emb, eps=cfg.w_eps, track_running_stats=False)
        head_mse = get_head(out_size, cfg)
        params += list(head_mse.parameters())
    if cfg.nce:
        head_nce = get_head(out_size, cfg, bn_last=True)
        params += list(head_nce.parameters())

    optimizer = optim.Adam(
        params, lr=cfg.lr, betas=(cfg.adam_b0, 0.999), weight_decay=cfg.adam_l2
    )
    if cfg.lr_step == "cos":
        scheduler = CosineAnnealingWarmRestarts(
            optimizer, T_0=cfg.T0, T_mult=cfg.Tmult, eta_min=cfg.eta_min
        )
    elif cfg.lr_step == "step":
        scheduler = MultiStepLR(optimizer, milestones=cfg.drop, gamma=cfg.drop_gamma)

    if cfg.fname is not None:
        cpoint = torch.load(cfg.fname)
        model.load_state_dict(cpoint["model"])
        optimizer.load_state_dict(cpoint["optimizer"])
        if cfg.w_mse:
            head_mse.load_state_dict(cpoint["head_mse"])
        if cfg.nce:
            head_nce.load_state_dict(cpoint["head_nce"])

    def save(fname):
        torch.save(
            {
                "model": model.state_dict(),
                "head_mse": head_mse.state_dict() if cfg.w_mse else None,
                "head_nce": head_nce.state_dict() if cfg.nce else None,
                "optimizer": optimizer.state_dict(),
            },
            fname,
        )

    bs = cfg.bs
    lr_warmup = 0 if cfg.lr_warmup else 500
    cudnn.benchmark = True
    for ep in trange(cfg.epoch, position=0):
        loss_ep = []
        iters = len(ds.train)
        for n_iter, (samples, _) in enumerate(tqdm(ds.train, position=1)):
            if lr_warmup < 500:
                lr_scale = (lr_warmup + 1) / 500
                for pg in optimizer.param_groups:
                    pg["lr"] = cfg.lr * lr_scale
                lr_warmup += 1

            optimizer.zero_grad()
            loss = 0
            h = [model(x.cuda(non_blocking=True)) for x in samples]

            if cfg.nce:
                z_full = head_nce(h)
                for i in range(len(samples) - 1):
                    for j in range(i + 1, len(samples)):
                        x0 = z_full[i * bs : (i + 1) * bs]
                        x1 = z_full[j * bs : (j + 1) * bs]
                        loss += contrastive_loss(x0, x1, tau=cfg.tau, norm=cfg.norm)

            if cfg.w_mse:
                h = head_mse(torch.cat(h))
                h = [whitening(h[i * bs : (i + 1) * bs]) for i in range(len(samples))]
                for i in range(len(samples) - 1):
                    for j in range(i + 1, len(samples)):
                        loss += norm_mse_loss(h[i], h[j])

            loss /= sum(range(len(samples)))
            loss.backward()
            optimizer.step()
            loss_ep.append(loss.item())
            if cfg.lr_step == "cos" and lr_warmup >= 500:
                scheduler.step(ep + n_iter / iters)

        if cfg.lr_step == "step":
            scheduler.step()

        if (ep + 1) % cfg.eval_every == 0:
            acc_knn = eval_knn(model, out_size, ds.clf, ds.test, cfg.knn)
            acc = eval_sgd(model, out_size, ds.clf, ds.test, 500)
            wandb.log({"acc": acc, "acc_knn": acc_knn}, commit=False)
            model.train()

        if (ep + 1) % 100 == 0:
            save(f"data/{cfg.dataset}_{ep}.pt")

        wandb.log({"loss": np.mean(loss_ep), "ep": ep})

    if cfg.save_model:
        fname = f"data/{wrun.id}.pt"
        save(fname)
        wandb.save(fname)

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
from eval_knn import eval_knn
from datasets import get_ds


def contrastive_loss(x0, x1, tau, norm=True):
    # https://github.com/google-research/simclr/blob/master/objective.py
    bsize = x0.shape[0]
    target = torch.arange(bsize).cuda()
    eye_mask = torch.eye(bsize).cuda() * 1e9
    if norm:
        x0 = normalize(x0, p=2, dim=1)
        x1 = normalize(x1, p=2, dim=1)
    logits00 = x0 @ x0.t() / tau - eye_mask
    logits11 = x1 @ x1.t() / tau - eye_mask
    logits01 = x0 @ x1.t() / tau
    logits10 = x1 @ x0.t() / tau
    return (
        cross_entropy(torch.cat([logits01, logits00], dim=1), target)
        + cross_entropy(torch.cat([logits10, logits11], dim=1), target)
    ) / 2


if __name__ == "__main__":
    cfg = get_cfg()
    wrun = wandb.init(project="white_ss", config=cfg)

    ds = get_ds(cfg.dataset)(cfg.bs, cfg)
    model, out_size = get_model(cfg.arch, cfg.dataset)
    params = list(model.parameters())
    head = get_head(out_size, cfg)
    params += list(head.parameters())

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
        head.load_state_dict(cpoint["head"])
        optimizer.load_state_dict(cpoint["optimizer"])

    def save(fname):
        torch.save(
            {
                "model": model.state_dict(),
                "head": head.state_dict(),
                "optimizer": optimizer.state_dict(),
            },
            fname,
        )

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
            h_len = len(h[0])
            h = torch.cat(h)
            z = head(h)

            if cfg.w_mse:
                for i in range(len(samples) - 1):
                    for j in range(i + 1, len(samples)):
                        x0 = z[i * h_len : (i + 1) * h_len]
                        x1 = z[j * h_len : (j + 1) * h_len]
                        loss += mse_loss(x0, x1)
                loss /= sum(range(len(samples) - 1))

            if cfg.nce:
                assert len(samples) == 2
                x0, x1 = z[:h_len], z[h_len:]
                loss += contrastive_loss(x0, x1, cfg.tau, cfg.norm)

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

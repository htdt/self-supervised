import argparse
from torchvision import models
from datasets import DS_LIST
from methods import METHOD_LIST


def get_cfg():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--method", type=str, choices=METHOD_LIST, default="w_mse")
    parser.add_argument("--byol_tau", type=float, default=0.99)
    parser.add_argument("--num_samples", type=int, default=4)

    parser.add_argument("--cj0", type=float, default=0.4)
    parser.add_argument("--cj1", type=float, default=0.4)
    parser.add_argument("--cj2", type=float, default=0.4)
    parser.add_argument("--cj3", type=float, default=0.1)
    parser.add_argument("--cj_p", type=float, default=0.8)
    parser.add_argument("--gs_p", type=float, default=0.1)
    parser.add_argument("--crop_s0", type=float, default=0.2)
    parser.add_argument("--crop_s1", type=float, default=1.0)
    parser.add_argument("--crop_r0", type=float, default=0.75)
    parser.add_argument("--crop_r1", type=float, default=(4 / 3))
    parser.add_argument("--hf_p", type=float, default=0.5)

    parser.add_argument("--no_lr_warmup", dest="lr_warmup", action="store_false")
    parser.add_argument("--no_add_bn", dest="add_bn", action="store_false")
    parser.add_argument("--knn", type=int, default=5)
    parser.add_argument("--fname", type=str)
    parser.add_argument(
        "--lr_step", type=str, choices=["cos", "step", "none"], default="step"
    )
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument("--eta_min", type=float, default=0)
    parser.add_argument("--adam_l2", type=float, default=1e-6)
    parser.add_argument("--T0", type=int)
    parser.add_argument("--Tmult", type=int, default=1)
    parser.add_argument("--w_eps", type=float, default=0)
    parser.add_argument("--head_layers", type=int, default=2)
    parser.add_argument("--head_size", type=int, default=1024)

    parser.add_argument("--w_size", type=int, default=128)
    parser.add_argument(
        "--w_iter",
        type=int,
        default=1,
        help="iterations for whitening matrix estimation",
    )

    parser.add_argument(
        "--no_norm", dest="norm", action="store_false", help="don't normalize latents",
    )
    parser.add_argument("--tau", type=float, default=0.5, help="InfoNCE temperature")

    parser.add_argument("--epoch", type=int, default=200, help="total epoch number")
    parser.add_argument(
        "--eval_every_drop",
        type=int,
        default=5,
        help="how often to evaluate after drop",
    )
    parser.add_argument(
        "--eval_every", type=int, default=20, help="how often to evaluate"
    )
    parser.add_argument("--emb", type=int, default=64, help="embedding size")
    parser.add_argument("--bs", type=int, default=256, help="batch size")
    parser.add_argument(
        "--drop",
        type=int,
        nargs="*",
        default=[50, 25],
        help="milestones for learning rate decay (0 = last epoch)",
    )
    parser.add_argument(
        "--drop_gamma",
        type=float,
        default=0.2,
        help="multiplicative factor of learning rate decay",
    )
    parser.add_argument(
        "--arch",
        type=str,
        choices=dir(models) + ["DIM32", "DIM64"],
        default="resnet18",
        help="encoder architecture",
    )
    parser.add_argument("--dataset", type=str, choices=DS_LIST, default="cifar10")
    return parser.parse_args()

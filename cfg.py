import argparse
from torchvision import models
from datasets import DS_LIST


def get_cfg():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--w_mse", action="store_true", help="use W-MSE loss")
    parser.add_argument(
        "--w_iter",
        type=int,
        default=1,
        help="iterations for whitening matrix estimation",
    )
    parser.add_argument(
        "--w_slice",
        type=int,
        default=1,
        help="number of batch slices for whitening matrix estimation",
    )

    parser.add_argument("--nce", action="store_true", help="use InfoNCE loss")
    parser.add_argument(
        "--no_norm",
        dest="norm",
        action="store_false",
        help="don't normalize InfoNCE latents",
    )
    parser.add_argument("--tau", type=float, default=0.5, help="InfoNCE temperature")

    parser.add_argument(
        "--linear_head", action="store_true", help="use linear head instead of MLP"
    )
    parser.add_argument("--epoch", type=int, default=200, help="total epoch number")
    parser.add_argument(
        "--eval_every", type=int, default=20, help="how often to evaluate"
    )
    parser.add_argument("--emb", type=int, default=32, help="embedding size")
    parser.add_argument(
        "--l2", type=float, default=0, help="weight decay regularization"
    )
    parser.add_argument("--bs", type=int, default=256, help="batch size")
    parser.add_argument(
        "--drop",
        type=int,
        nargs="*",
        default=[],
        help="milestones for learning rate decay",
    )
    parser.add_argument(
        "--drop_gamma",
        type=float,
        default=0.1,
        help="multiplicative factor of learning rate decay",
    )
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument(
        "--arch",
        type=str,
        choices=dir(models) + ["DIM32", "DIM64"],
        default="resnet18",
        help="encoder architecture",
    )
    parser.add_argument("--dataset", type=str, choices=DS_LIST, default="cifar10")
    return parser.parse_args()

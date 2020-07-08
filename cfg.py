import argparse
from torchvision import models
from datasets import DS_LIST


def get_cfg():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--cj0", type=float, default=0.6)
    parser.add_argument("--cj1", type=float, default=0.6)
    parser.add_argument("--cj2", type=float, default=0.6)
    parser.add_argument("--cj3", type=float, default=0.2)
    parser.add_argument("--cj_p", type=float, default=0.5)
    parser.add_argument("--gs_p", type=float, default=0.1)
    parser.add_argument("--crop_s0", type=float, default=0.2)
    parser.add_argument("--crop_s1", type=float, default=1.0)
    parser.add_argument("--crop_r0", type=float, default=0.75)
    parser.add_argument("--crop_r1", type=float, default=1.33333)
    parser.add_argument("--hf_p", type=float, default=0.5)

    parser.add_argument("--lr", type=float, default=0.1, help="learning rate")
    parser.add_argument("--eta_min", type=float, default=1e-4)
    parser.add_argument("--T0", type=int, default=10)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument("--w_eps", type=float, default=0)
    parser.add_argument("--head_layers", type=int, default=2)
    parser.add_argument(
        "--method", type=str, choices=["cholesky", "zca"], default="cholesky"
    )

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

    parser.add_argument("--epoch", type=int, default=200, help="total epoch number")
    parser.add_argument(
        "--eval_every", type=int, default=20, help="how often to evaluate"
    )
    parser.add_argument("--emb", type=int, default=32, help="embedding size")
    parser.add_argument("--bs", type=int, default=256, help="batch size")
    # parser.add_argument(
    #     "--drop",
    #     type=int,
    #     nargs="*",
    #     default=[],
    #     help="milestones for learning rate decay",
    # )
    # parser.add_argument(
    #     "--drop_gamma",
    #     type=float,
    #     default=0.1,
    #     help="multiplicative factor of learning rate decay",
    # )
    parser.add_argument(
        "--arch",
        type=str,
        choices=dir(models) + ["DIM32", "DIM64"],
        default="resnet50",
        help="encoder architecture",
    )
    parser.add_argument("--dataset", type=str, choices=DS_LIST, default="cifar10")
    return parser.parse_args()

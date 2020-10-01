import argparse
import torch
from model import get_model
from datasets import get_ds, DS_LIST
from eval.sgd import eval_sgd
from eval.knn import eval_knn
from eval.lbfgs import eval_lbfgs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--arch", type=str, default="resnet18")
    parser.add_argument(
        "--clf", type=str, default="sgd", choices=["sgd", "knn", "lbfgs"]
    )
    parser.add_argument("--dataset", type=str, choices=DS_LIST, default="cifar10")
    parser.add_argument("--fname", type=str, help="load model from file")
    cfg = parser.parse_args()

    model, out_size = get_model(cfg.arch, cfg.dataset)
    if cfg.fname is None:
        print("evaluating random model")
    else:
        model.load_state_dict(torch.load(cfg.fname))

    ds = get_ds(cfg.dataset)(None, cfg)
    if cfg.clf == "sgd":
        acc = eval_sgd(model, out_size, ds.clf, ds.test, 500)
    if cfg.clf == "knn":
        acc = eval_knn(model, out_size, ds.clf, ds.test, 5)
    elif cfg.clf == "lbfgs":
        acc = eval_lbfgs(model, ds.clf, ds.test)
    print(acc)

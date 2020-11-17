import argparse
import torch
from model import get_model
from datasets import get_ds, DS_LIST
from eval.sgd import eval_sgd
from eval.knn import eval_knn
from eval.lbfgs import eval_lbfgs
from eval.get_data import get_data


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
    model.cuda().eval()
    if cfg.fname is None:
        print("evaluating random model")
    else:
        model.load_state_dict(torch.load(cfg.fname))

    ds = get_ds(cfg.dataset)(None, cfg)
    device = "cpu" if cfg.clf == "lbfgs" else "cuda"
    x_train, y_train = get_data(model, ds.clf, out_size, device)
    x_test, y_test = get_data(model, ds.test, out_size, device)

    if cfg.clf == "sgd":
        acc = eval_sgd(x_train, y_train, x_test, y_test, 500)
    if cfg.clf == "knn":
        acc = eval_knn(x_train, y_train, x_test, y_test)
    elif cfg.clf == "lbfgs":
        acc = eval_lbfgs(x_train, y_train, x_test, y_test)
    print(acc)

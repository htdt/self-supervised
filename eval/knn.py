import torch
from .get_data import get_data


def eval_knn(model, output_size, loader_clf, loader_test, k=5):
    """ k-nearest neighbors classifier accuracy """
    model.eval()
    x_train, y_train = get_data(model, loader_clf, output_size, "cpu")
    x_test, y_test = get_data(model, loader_test, output_size, "cpu")

    d = torch.cdist(x_test, x_train)
    topk = torch.topk(d, k=k, dim=1, largest=False)
    labels = y_train[topk.indices]
    pred = torch.empty_like(y_test)
    for i in range(len(labels)):
        x = labels[i].unique(return_counts=True)
        pred[i] = x[0][x[1].argmax()]

    acc = (pred == y_test).float().mean().cpu().item()
    del x_train, y_train, x_test, y_test, d, topk, labels, pred
    return acc

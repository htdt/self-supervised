import torch


def get_data(model, loader, output_size, device):
    xs = torch.empty(
        len(loader), loader.batch_size, output_size, dtype=torch.float32, device=device
    )
    ys = torch.empty(len(loader), loader.batch_size, dtype=torch.long, device=device)
    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            x = x.cuda()
            xs[i] = model(x).to(device)
            ys[i] = y.to(device)
    xs = xs.view(-1, output_size)
    ys = ys.view(-1)
    return xs, ys


def eval_knn(model, output_size, loader_clf, loader_test, k=5):
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

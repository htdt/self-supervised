import torch
from sklearn.linear_model import LogisticRegression


def get_data(model, loader):
    x_list, y_list = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.cuda()
            x_list.append(model(x).cpu())
            y_list.append(y)
    return torch.cat(x_list), torch.cat(y_list)


def eval_lbfgs(model, loader_clf, loader_test):
    """ linear classifier accuracy (lbfgs method) """
    model.eval()
    clf = LogisticRegression(
        random_state=1337, solver="lbfgs", max_iter=1000, n_jobs=-1
    )
    clf.fit(*get_data(model, loader_clf))
    x_test, y_test = get_data(model, loader_test)
    pred = clf.predict(x_test)
    return (torch.tensor(pred) == y_test).float().mean()

import torch
from sklearn.linear_model import LogisticRegression


def eval_lbfgs(x_train, y_train, x_test, y_test):
    """ linear classifier accuracy (lbfgs method) """
    clf = LogisticRegression(
        random_state=1337, solver="lbfgs", max_iter=1000, n_jobs=-1
    )
    clf.fit(x_train, y_train)
    pred = clf.predict(x_test)
    return (torch.tensor(pred) == y_test).float().mean()

from tqdm import trange
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
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
    model.eval()
    fc_prev = model.fc
    model.fc = nn.Identity()
    clf = LogisticRegression(
        random_state=1337, solver='lbfgs', max_iter=1000, n_jobs=-1)
    clf.fit(*get_data(model, loader_clf))
    x_test, y_test = get_data(model, loader_test)
    pred = clf.predict(x_test)
    acc = (torch.tensor(pred) == y_test).float().mean()
    wandb.log({'acc': acc})
    model.fc = fc_prev


def eval_sgd(model, output_size, loader_clf, loader_test, epoch):
    milestones = [epoch - i * 20 for i in range(3, 0, -1)]
    model.eval()
    fc_prev = model.fc
    model.fc = nn.Identity()
    clf = nn.Linear(output_size, 10)
    clf.cuda()
    clf.train()
    optimizer = optim.Adam(clf.parameters(), lr=5e-3)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones)
    criterion = nn.CrossEntropyLoss()

    x_test, y_test = get_data(model, loader_test)
    x_test, y_test = x_test.cuda(), y_test.cuda()
    for ep in trange(epoch):
        loss_ep = []
        for x, y in loader_clf:
            with torch.no_grad():
                x = model(x.cuda())
            y = y.cuda()
            optimizer.zero_grad()
            loss = criterion(clf(x), y)
            loss.backward()
            optimizer.step()
            loss_ep.append(loss.item())

        if (ep + 1) % 5 == 0:
            clf.eval()
            with torch.no_grad():
                y_pred = clf(x_test)
            acc = (y_pred.argmax(1) == y_test).float().mean().cpu().item()
            wandb.log({'acc': acc}, commit=False)
            clf.train()

        wandb.log({'loss': np.mean(loss_ep), 'ep': ep})
        scheduler.step()
    model.fc = fc_prev

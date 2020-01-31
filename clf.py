from tqdm import trange
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn


def get_acc(model, loader_clf, loader_test):
    model.eval()
    fc_orig = model.fc
    model.fc = nn.Identity()

    clf = nn.Linear(512, 10)
    clf.cuda()
    clf.train()
    optimizer = optim.Adam(clf.parameters(), lr=5e-3)
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[190, 210, 230])
    criterion = nn.CrossEntropyLoss()

    for ep in trange(250):
        for x, y in loader_clf:
            with torch.no_grad():
                x = model(x.cuda())
            y = y.cuda()
            optimizer.zero_grad()
            loss = criterion(clf(x), y)
            loss.backward()
            optimizer.step()
        scheduler.step()

    clf.eval()
    acc = []
    with torch.no_grad():
        for x, y in loader_test:
            bs, ncrops, c, h, w = x.shape
            x = x.cuda().view(bs * ncrops, c, h, w)
            y_pred = clf(model(x))
            y_pred = y_pred.view(bs, ncrops, 10).mean(1)
            acc_cur = (y_pred.argmax(1).cpu() == y).float().mean().item()
            acc.append(acc_cur)

    model.fc = fc_orig
    return np.mean(acc)

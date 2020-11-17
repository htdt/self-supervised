import torch
import torch.nn as nn
import torch.optim as optim


def eval_sgd(x_train, y_train, x_test, y_test, epoch=300):
    """ linear classifier accuracy (sgd) """
    lr_start, lr_end = 1e-2, 1e-6
    gamma = (lr_end / lr_start) ** (1 / epoch)
    output_size = x_train.shape[1]
    num_class = y_train.max().item() + 1
    clf = nn.Linear(output_size, num_class)
    clf.cuda()
    clf.train()
    optimizer = optim.Adam(clf.parameters(), lr=lr_start, weight_decay=5e-6)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    criterion = nn.CrossEntropyLoss()

    for ep in range(epoch):
        perm = torch.randperm(len(x_train)).view(-1, 1000)
        for idx in perm:
            optimizer.zero_grad()
            criterion(clf(x_train[idx]), y_train[idx]).backward()
            optimizer.step()
        scheduler.step()

    clf.eval()
    with torch.no_grad():
        y_pred = clf(x_test)
    acc = (y_pred.argmax(1) == y_test).float().mean().cpu().item()
    del clf
    return acc

from tqdm import trange
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn


def get_acc(model, loader_clf, loader_test, epoch):
    model.eval()
    fc_orig = model.fc
    model.fc = nn.Identity()
    z = torch.zeros(1, 3, 32, 32, device='cuda')
    with torch.no_grad():
        output_size = model(z).shape[-1]
        print('model output', output_size)

    clf = nn.Linear(output_size, 10)
    clf.cuda()
    clf.train()
    optimizer = optim.Adam(clf.parameters(), lr=5e-3)
    milestones = [epoch - 10] if epoch != 250 else [190, 210, 230]
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=milestones)
    criterion = nn.CrossEntropyLoss()

    for ep in trange(epoch):
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
            x = x.cuda()
            if len(x.shape) == 5:
                bs, ncrops, c, h, w = x.shape
                x = x.view(bs * ncrops, c, h, w)
            else:
                ncrops = None
            y_pred = clf(model(x))
            if ncrops is not None:
                y_pred = y_pred.view(bs, ncrops, 10).mean(1)
            acc_cur = (y_pred.argmax(1).cpu() == y).float().mean().item()
            acc.append(acc_cur)

    model.fc = fc_orig
    return np.mean(acc)

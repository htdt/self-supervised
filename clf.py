from tqdm import trange
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import wandb


def get_acc(model, loader_clf, loader_test, epoch, milestones, lr):
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
    optimizer = optim.Adam(clf.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=milestones)
    criterion = nn.CrossEntropyLoss()

    x_test, y_test = [], []
    with torch.no_grad():
        for x, y in loader_test:
            x = x.cuda()
            x_test.append(model(x))
            y_test.append(y)
    x_test = torch.cat(x_test)
    y_test = torch.cat(y_test).cuda()

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

    model.fc = fc_orig

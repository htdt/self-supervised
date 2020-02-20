from tqdm import trange, tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wandb


def get_data(model, loader, output_size, device):
    xs = torch.empty(len(loader), loader.batch_size, output_size,
                     dtype=torch.float32, device=device)
    ys = torch.empty(len(loader), loader.batch_size,
                     dtype=torch.long, device=device)
    with torch.no_grad():
        for i, (x, y) in enumerate(tqdm(loader)):
            x = x.cuda()
            xs[i] = model(x).to(device)
            ys[i] = y.to(device)
    xs = xs.view(-1, output_size)
    ys = ys.view(-1)
    return xs, ys


def eval_sgd(model, output_size, loader_clf, loader_test):
    model.eval()
    x_train, y_train = get_data(model, loader_clf, output_size, 'cpu')
    x_test, y_test = get_data(model, loader_test, output_size, 'cpu')
    torch.save({
        'x_train': x_train,
        'y_train': y_train,
        'x_test': x_test,
        'y_test': y_test,
    }, 'data/emb.pt')

    del model
    torch.cuda.empty_cache()
    x_train, y_train = x_train.cuda(), y_train.cuda()
    x_test, y_test = x_test.cuda(), y_test.cuda()

    epoch = 300
    lr_start, lr_end = 1e-2, 1e-4
    gamma = (lr_end / lr_start) ** (1 / epoch)
    clf = nn.Linear(output_size, 1000)
    clf.cuda()
    clf.train()
    optimizer = optim.Adam(clf.parameters(), lr=lr_start, weight_decay=5e-6)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    criterion = nn.CrossEntropyLoss()

    for ep in trange(epoch):
        loss_ep = []
        perm = torch.randperm(len(x_train)).view(-1, 1000)
        for idx in perm:
            optimizer.zero_grad()
            loss = criterion(clf(x_train[idx]), y_train[idx])
            loss.backward()
            optimizer.step()
            loss_ep.append(loss.item())

        wandb.log({'loss': np.mean(loss_ep)})
        scheduler.step()

        if (ep + 1) % 5 == 0:
            clf.eval()
            with torch.no_grad():
                y_pred = clf(x_test)
            acc = (y_pred.argmax(1) == y_test).float().mean().cpu().item()
            wandb.log({'acc': acc, 'ep': ep})
            clf.train()

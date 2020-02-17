from tqdm import trange, tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wandb


def eval_sgd_chunk(model, output_size, loader_clf, loader_test):
    num_chunks = 12
    num_ds_ep = 2
    num_mini_ep = 30
    num_steps = num_chunks * num_ds_ep * num_mini_ep

    model.eval()
    clf = nn.Linear(output_size, 1000)
    clf.cuda()
    clf.train()
    optimizer = optim.Adam(clf.parameters(), lr=5e-3)
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, [num_steps // 2, num_steps * 3 // 4])
    criterion = nn.CrossEntropyLoss()

    chunk = len(loader_clf) // num_chunks + 1
    x_train = torch.empty(chunk, 1000, output_size,
                          dtype=torch.float32, device='cuda')
    y_train = torch.empty(chunk, 1000,
                          dtype=torch.long, device='cuda')
    x_test = torch.empty(len(loader_test), 1000, output_size,
                         dtype=torch.float32, device='cuda')
    y_test = torch.empty(len(loader_test), 1000,
                         dtype=torch.long, device='cuda')

    with torch.no_grad():
        for i, (x, y) in enumerate(tqdm(loader_test)):
            x = x.cuda()
            x_test[i] = model(x)
            y_test[i] = y
    x_test = x_test.view(-1, output_size)
    y_test = y_test.view(-1)
    mini_ep = 0

    for ds_epoch in range(num_ds_ep):
        for i, (x, y) in enumerate(tqdm(loader_clf)):
            x = x.cuda()
            with torch.no_grad():
                x_train[i % chunk] = model(x)
            y_train[i % chunk] = y

            if (i + 1) % chunk == 0 or (i + 1) == len(loader_clf):
                x_train = x_train.view(-1, output_size)
                y_train = y_train.view(-1)
                len_x = (i % chunk + 1) * 1000

                for sgd_ep in trange(num_mini_ep):
                    loss_ep = []
                    perm = torch.randperm(len_x).view(-1, 1000)
                    for idx in perm:
                        x, y = x_train[idx], y_train[idx]
                        optimizer.zero_grad()
                        loss = criterion(clf(x), y)
                        loss.backward()
                        optimizer.step()
                        loss_ep.append(loss.item())

                    wandb.log({'loss': np.mean(loss_ep)})
                    scheduler.step()

                mini_ep += 30
                clf.eval()
                with torch.no_grad():
                    y_pred = clf(x_test)
                acc = (y_pred.argmax(1) == y_test).float().mean().cpu().item()
                wandb.log({'acc': acc, 'mini_ep': mini_ep})
                clf.train()
                x_train = x_train.view(chunk, 1000, output_size)
                y_train = y_train.view(chunk, 1000)

import argparse
import torch
import torch.nn as nn
from torchvision import models
from dataset import get_loader_clf, get_loader_test
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


def get_acc(model, loader_clf, loader_test):
    model.eval()
    model.fc = nn.Identity()
    clf = LogisticRegression(
        random_state=1337, solver='lbfgs', max_iter=1000, n_jobs=-1, verbose=1)
    clf.fit(*get_data(model, loader_clf))
    x_test, y_test = get_data(model, loader_test)
    pred = clf.predict(x_test)
    acc = (torch.tensor(pred) == y_test).float().mean()
    wandb.log({'acc': acc})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--emb', type=int, default=32)
    parser.add_argument(
        '--arch', type=str, choices=dir(models), default='resnet18')
    parser.add_argument('--fname', type=str, required=True)
    cfg = parser.parse_args()
    cfg.mode = 'clf_lbfgs'
    wandb.init(project="white_ss", config=cfg)

    model = getattr(models, cfg.arch)(num_classes=cfg.emb)
    model.conv1 = nn.Conv2d(
        3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.cuda()
    checkpoint = torch.load(cfg.fname)
    model.load_state_dict(checkpoint['model'])

    get_acc(model, get_loader_clf(), get_loader_test())

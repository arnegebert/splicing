import torch.nn.functional as F
import torch


def nll_loss(prediction, target):
    return F.nll_loss(prediction, target)


def mse_loss(prediction, target):
    return F.mse_loss(prediction, target)


def l1_loss(prediction, target):
    return F.l1_loss(prediction, target.view(-1, 1))

def cross_entropy(prediction, target):
    return F.cross_entropy(prediction, target >= 0.8)
    # if target >= 0.8:
    #     return F.cross_entropy(prediction, torch.tensor(1))
    # else:
    #     return F.cross_entropy(prediction, torch.tensor(0))

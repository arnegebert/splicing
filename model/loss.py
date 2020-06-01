import torch.nn.functional as F


def nll_loss(prediction, target):
    return F.nll_loss(prediction, target)


def mse_loss(prediction, target):
    return F.mse_loss(prediction, target)


def l1_loss(prediction, target):
    return F.l1_loss(prediction, target.view(-1, 1))

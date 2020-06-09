import torch
from sklearn import metrics

def accuracy(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
    return correct / len(target)


def top_k_acc(output, target, k=3):
    with torch.no_grad():
        pred = torch.topk(output, k, dim=1)[1]
        assert pred.shape[0] == len(target)
        correct = 0
        for i in range(k):
            correct += torch.sum(pred[:, i] == target).item()
    return correct / len(target)

def auc(output, target):
    with torch.no_grad():
        return metrics.roc_auc_score(target.cpu(), output.cpu())

def r2(output, target):
    with torch.no_grad():
        return metrics.r2_score(target.cpu(), output.cpu())
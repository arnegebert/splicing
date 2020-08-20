import torch
from sklearn import metrics
from sklearn.metrics import roc_curve
import numpy as np

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

def f1(output, target):
    with torch.no_grad():
        return metrics.f1_score(target.cpu(), output.cpu())

def recall(output, target):
    with torch.no_grad():
        return metrics.recall_score(target.cpu(), output.cpu())

def precision(output, target):
    with torch.no_grad():
        return metrics.precision_score(target.cpu(), output.cpu())

def auc(output, target):
    with torch.no_grad():
        return metrics.roc_auc_score(target.cpu(), output.cpu())

def auc_single(output_and_target):
    output, target = output_and_target
    # return auc(output, target)
    output, target = np.array(output).flatten(), np.array(target)
    return metrics.roc_auc_score(target, output)

# same as auc
def auc2(output, target):
    with torch.no_grad():
        fpr, tpr, thresholds = roc_curve(target.cpu(), output.cpu())
        return metrics.auc(fpr, tpr)

def r2(output, target):
    with torch.no_grad():
        return metrics.r2_score(target.cpu(), output.cpu())
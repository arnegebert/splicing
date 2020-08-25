import functools
import json
import time
import numpy as np

import pandas as pd
from pathlib import Path
from itertools import repeat
from collections import OrderedDict
import math

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc


def compute_relative_performance_change_auc(value_bef, value_after):
    return (value_after-value_bef)/(value_after-0.5)

def overlap(start, end, start2, end2):
    return not (start > end2 or end < start2)

def plot_and_save_roc(save_dir, *args):
    plt.cla()
    plt.style.use('seaborn')
    colors = ['orange', 'green', 'blue']
    for (((pred, target), label), color) in zip(args, colors):
        fpr, tpr, _ = roc_curve(target, pred)
        auc_val = auc(fpr, tpr)
        plt.plot(fpr, tpr, linestyle='--', color=color, label=f'{label} PSI events (AUC={auc_val:0.2f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    # plt.title('ROC curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')

    plt.legend(loc='best')
    plt.savefig(f'{save_dir}/ROC.png', dpi=300, bbox_inches='tight')
    # plt.show()

def save_pred_and_target(log_dir, pred_target_all, pred_target_low, pred_target_high):
    # pred_target_all = (pred_all, target_all)
    pred_target_all = np.concatenate((pred_target_all[0].flatten(), pred_target_all[1]))
    pred_target_low = np.concatenate((pred_target_low[0].flatten(), pred_target_low[1]))
    pred_target_high = np.concatenate((pred_target_high[0].flatten(), pred_target_high[1]))
    np.save(f'{log_dir}/pred_and_target_all.npy', pred_target_all)
    np.save(f'{log_dir}/pred_and_target_low.npy', pred_target_low)
    np.save(f'{log_dir}/pred_and_target_high.npy', pred_target_high)

def timer(func):
    """Print the runtime of the decorated function"""
    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        start_time = time.perf_counter()  # 1
        value = func(*args, **kwargs)
        end_time = time.perf_counter()  # 2
        run_time = end_time - start_time  # 3
        print(f"Finished {func.__name__!r} in {run_time:.4f} secs")
        return value
    return wrapper_timer #  no "()" here, we need the object to
                         #  be returned.

def split_into_sentences(text, n):
    return [text[i * n:(i + 1) * n] for i in range(0, math.ceil(len(text) / n))]


def split_into_3_mers(sentence):
    words = []
    for i in range(1, len(sentence) - 1):
        words.append(sentence[i - 1:i + 2])
    return words

def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)

def read_json(fname):
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)

def write_json(content, fname):
    fname = Path(fname)
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)

def inf_loop(data_loader):
    ''' wrapper function for endless data loader. '''
    for loader in repeat(data_loader):
        yield from loader

class MetricTracker:
    def __init__(self, *keys, writer=None):
        self.writer = writer
        self.metrics = keys
        self._data = pd.DataFrame(index=keys, columns=['total', 'counts', 'average'])
        self.reset()
        
    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

    ''' updates saved metric from key, using value '''
    def update(self, key, value, n=1):
        if self.writer is not None:
            self.writer.add_scalar(key, value)
        self._data.total[key] += value * n
        self._data.counts[key] += n
        self._data.average[key] = self._data.total[key] / self._data.counts[key]

    def avg(self, key):
        return self._data.average[key]
    
    def result(self):
        return dict(self._data.average)

    def __contains__(self, item):
        return item in self.metrics
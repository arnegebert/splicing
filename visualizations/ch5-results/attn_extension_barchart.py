import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import numpy as np


def bar_charts():
    plt.style.use('seaborn')
    labels = ['no extension', 'heads', 'heads + no query', 'heads + conv',]
    means = [0.832,0.865, 0.852, 0.824]
    stds = [0.025, 0.011, 0.017, 0.010]

    x = np.arange(len(means))  # the label locations
    width = 0.7  # the width of the bars

    fig, ax = plt.subplots()
    rects = ax.bar(x, means, width, yerr=stds)

    ax.set_xticks(np.arange(len(means)))
    ax.set_ylim(0.5, 1)
    ax.set_ylabel('AUC')
    ax.set_xticklabels(labels)
    # ax.legend()

    def autolabel(rects, labels=None, stds=None):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for i, rect in enumerate(rects):
            label = labels[i] if labels else rect.get_height()
            height = rect.get_height()
            y_label = height + 0.02 if not stds else height + stds[i]*0.7 + 0.002
            ax.annotate(f'{label}',
                        xy=(rect.get_x() + rect.get_width() / 2, y_label),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')
    autolabel(rects, stds=stds)

    fig.tight_layout()
    plt.savefig('attn_extension_barcharts.png', dpi=300, bbox_inches='tight')
    plt.show()

def bar_charts_exhaustive():
    plt.style.use('seaborn')
    labels = ['no extension', 'conv', 'no query', 'no query + conv', 'heads', 'heads + conv',
              'heads + \nno query', 'heads + \nno query + conv']
    means = [0.832, 0.860, 0.851, 0.818, 0.865, 0.824, 0.852, 0.846]
    stds = [0.025, 0.021, 0.013, 0.019, 0.011, 0.010, 0.017, 0.017]

    x = np.arange(len(means))  # the label locations
    width = 0.7  # the width of the bars

    fig, ax = plt.subplots()
    rects = ax.bar(x, means, width, yerr=stds)

    ax.set_xticks(np.arange(len(means)))
    ax.set_ylim(0.5, 1)
    ax.set_ylabel('AUC')
    ax.set_xticklabels(labels)
    # ax.legend()

    def autolabel(rects, labels=None):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for i, rect in enumerate(rects):
            label = labels[i] if labels else rect.get_height()
            height = rect.get_height()
            if label == '1/20,000': continue
            ax.annotate(f'{label}',
                        xy=(rect.get_x() + rect.get_width() / 2, height + 0.02),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')
    autolabel(rects)

    fig.tight_layout()
    plt.savefig('attn_extension_barcharts.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    bar_charts()
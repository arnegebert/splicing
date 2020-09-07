import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import numpy as np

def bar_charts():
    plt.style.use('seaborn')
    # can be further fine-tuned; either 0.5 from start or I include baseline model
    labels = ['DSC', 'D2V', 'RASC']
    all_means = [0.244, 0.120, 0.486, ]
    low_means = [0.340, 0.155, 0.595, ]
    high_means = [-77.980/100, -57.523/100, -25.470/100,]

    all_stds = [0.036, 0.009, 0.067]
    low_stds = [0.039, 0.010, 0.075]
    high_stds = [13.631/100, 2.771/100, 5.717/100]

    x = np.arange(len(all_means))  # the label locations
    width = 0.2  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width, all_means, width, yerr=all_stds, label='AllPSI')
    rects2 = ax.bar(x, low_means, width, yerr=low_stds, label='LowPSI')
    rects3 = ax.bar(x + width, high_means, width, yerr=high_stds, label='HighPSI')

    # ax.set_title('AUC and inverse number of weights by model')
    ax.set_xticks(np.arange(len(labels)))
    # ax.set_ylim(0.5, 1)
    ax.set_ylabel('R2')
    ax.set_xticklabels(labels)
    ax.legend()

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
    autolabel(rects1, stds=all_stds)
    autolabel(rects2, stds=low_stds)
    autolabel(rects3, stds=high_stds)

    fig.tight_layout()
    plt.savefig('regression_barcharts.png', dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    bar_charts()

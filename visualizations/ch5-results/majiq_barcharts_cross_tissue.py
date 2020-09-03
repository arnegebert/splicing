import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import numpy as np

def bar_charts():
    plt.style.use('seaborn')
    labels = ['DSC', 'D2V', 'RASC']
    same_means = [0.822, 0.720, 0.875]
    diff_lib_means = [0.856, 0.728, 0.903]
    diff_indv_means = [0.854, 0.728, 0.900]
    diff_tissue_means = [0.857, 0.732, 0.904]

    same_stds = [0.013, 0.007, 0.014]
    diff_lib_stds = [0.012, 0.010, 0.009]
    diff_indv_stds = [0.013, 0.010, 0.012]
    diff_tissue_stds = [0.008, 0.003, 0.011]

    x = np.arange(len(labels))  # the label locations
    width = 0.2  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - 1.5*width, same_means, width, yerr=same_stds, label='iPSC cell, same lib')
    rects2 = ax.bar(x - 0.5*width, diff_lib_means, width, yerr=diff_lib_stds, label='iPSC cell, diff lib')
    rects3 = ax.bar(x + 0.5*width, diff_indv_means, width, yerr=diff_indv_stds, label='iPSC cell, diff indv')
    rects4 = ax.bar(x + 1.5*width, diff_tissue_means, width, yerr=diff_tissue_stds, label='neuron cell, diff indv')

    # ax.set_title('AUC and inverse number of weights by model')
    ax.set_xticks(np.arange(len(labels)))
    ax.set_ylim(0.5, 1)
    ax.set_ylabel('AUC')
    ax.set_xticklabels(labels)
    ax.legend()

    def autolabel(rects, labels=None, stds=None):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for i, rect in enumerate(rects):
            label = labels[i] if labels else rect.get_height()
            height = rect.get_height()
            y_label = height + 0.0075 if not stds else height + stds[i]*0.7 + 0.0075
            ax.annotate(f'{label}',
                        xy=(rect.get_x() + rect.get_width() / 2, y_label),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')
    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)
    autolabel(rects4)


    fig.tight_layout()
    plt.savefig('majiq_comparison_barcharts.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    bar_charts()
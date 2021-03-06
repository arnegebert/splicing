import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import numpy as np

def bar_charts_exons():
    plt.style.use('seaborn')
    with_extra_random_guessing_plot = False
    # can be further fine-tuned; either 0.5 from start or I include baseline model
    labels = ['DSC', 'D2V', 'RASC']
    brain_means = [0.664, 0.617, 0.645]
    cerebellum_means = [0.649, 0.610, 0.631]
    heart_means = [0.657, 0.604, 0.627]

    brain_stds = [0.011, 0.010, 0.022]
    cerebellum_stds = [0.008, 0.008, 0.029]
    heart_stds = [0.016, 0.009, 0.014]

    x = np.arange(len(brain_means))  # the label locations
    width = 0.2  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width, brain_means, width, yerr=brain_stds, label='Brain cortex')
    rects2 = ax.bar(x, cerebellum_means, width, yerr=cerebellum_stds, label='Cerebellum')
    rects3 = ax.bar(x + width, heart_means, width, yerr=heart_stds, label='Heart')

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
            if label == '1/20,000': continue
            y_label = height + 0.02 if not stds else height + stds[i]*0.7 + 0.002
            ax.annotate(f'{label}',
                        xy=(rect.get_x() + rect.get_width() / 2, y_label),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')
    autolabel(rects1, stds=brain_stds)
    autolabel(rects2, stds=cerebellum_stds)
    autolabel(rects3, stds=heart_stds)

    if with_extra_random_guessing_plot:
        labels.append('Random guessing')
        ax.bar(x - width, 0.5, width, color='gray')
        ax.bar(x, 0.5, width, color='gray')
        ax.bar(x + width, 0.5, width, color='gray')

        baseline_x = max(x) + 1
        width2 = 0.3

        rect4 = ax.bar(baseline_x, 0.5, width2, color='grey')
        autolabel(rect4)
        ax.axhline(y=0.5, color='gray')

    fig.tight_layout()
    plt.savefig('gtex_exon_barcharts.png', dpi=300, bbox_inches='tight')
    plt.show()


def bar_charts_juncs():
    plt.style.use('seaborn')
    with_extra_random_guessing_plot = False
    # can be further fine-tuned; either 0.5 from start or I include baseline model
    labels = ['DSC', 'D2V', 'RASC']
    brain_means = [0.699, 0.671, 0.810]
    cerebellum_means = [0.704, 0.673, 0.808]
    heart_means = [0.699, 0.677, 0.807]

    brain_stds = [0.006, 0.003, 0.012]
    cerebellum_stds = [0.006, 0.004, 0.008]
    heart_stds = [0.008, 0.005, 0.013]

    x = np.arange(len(brain_means))  # the label locations
    width = 0.2  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width, brain_means, width, yerr=brain_stds, label='Brain cortex')
    rects2 = ax.bar(x, cerebellum_means, width, yerr=cerebellum_stds, label='Cerebellum')
    rects3 = ax.bar(x + width, heart_means, width, yerr=heart_stds, label='Heart')

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
            if label == '1/20,000': continue
            y_label = height + 0.02 if not stds else height + stds[i]*0.7 + 0.002
            ax.annotate(f'{label}',
                        xy=(rect.get_x() + rect.get_width() / 2, y_label),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')
    autolabel(rects1, stds=brain_stds)
    autolabel(rects2, stds=cerebellum_stds)
    autolabel(rects3, stds=heart_stds)

    if with_extra_random_guessing_plot:
        labels.append('Random guessing')
        ax.bar(x - width, 0.5, width, color='gray')
        ax.bar(x, 0.5, width, color='gray')
        ax.bar(x + width, 0.5, width, color='gray')

        baseline_x = max(x) + 1
        width2 = 0.3

        rect4 = ax.bar(baseline_x, 0.5, width2, color='grey')
        autolabel(rect4)
        ax.axhline(y=0.5, color='gray')

    fig.tight_layout()
    plt.savefig('gtex_junc_barcharts.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    bar_charts_exons()
    bar_charts_juncs()
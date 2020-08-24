import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import numpy as np

def bar_charts():
    plt.style.use('seaborn')

    labels = ['DSC (original)', 'MLP100', 'MLP20', 'MLPLinear']
    values = np.array([0.899, 0.900, 0.855, 0.505]) - 0.5
    vlabels = ['0.899', '0.900', '0.855', '0.505']

    use_inverse_ws = True
    if not use_inverse_ws:
        nweights = [0.5, 1/400, 1/2000, 1/2000]
        wlabels = ['20,000', '100', '20', '20']
    else:
        # progression to arrive at the final numbers
        # 1/20000, 1/100, 1/20, 1/20
        # 1/1000, 1/5, 1, 1 -- scale to 1
        # 1/2000, 1/10, 0.5, 0.5 -- scale to 0.5
        nweights = [1/2000, 1/10, 0.5, 0.5]
        wlabels = ['1/20,000', '1/100', '1/20', '1/20']

    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars
    width2 = 0.2

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width / 2, values, width, label='AUC')
    rects2 = ax.bar(x + width2 / 2, nweights, width2, label='Inverse number of weights')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    # ax.set_ylabel('Scores')
    # plt.yticks([], [])
    # ax.set_yticks([], minor=True)
    # plt.yticks([])
    # ax.axes.get_yaxis().set_visible(False)
    yticks = [item.get_text() for item in ax.get_yticklabels()]

    ax.set_yticklabels(['']*len(yticks))
    ax.set_title('AUC and inverse number of weights by model')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    def autolabel(rects, labels=None):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for i, rect in enumerate(rects):
            label = labels[i] if labels else rect.get_height()
            height = rect.get_height()
            if label == '1/20,000': continue
            ax.annotate(f'{label}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')
    autolabel(rects1, vlabels)
    autolabel(rects2, wlabels)
    dscwrect = rects2[0]
    dscwrect_height = dscwrect.get_height()
    ax.annotate(f'1/20,000',
                xy=(dscwrect.get_x() + dscwrect.get_width()-0.05, dscwrect_height),
                xytext=(0, 3),  # 3 points vertical offset
                textcoords="offset points",
                ha='center', va='bottom')
    # values = [0.899, 1, 0.900, 1/200, 0.855, 1/1000, 0.498, 1/1000]
    # xs = range(4)
    # labels = ['DSC (original)', 'MLP100', 'MLP20', 'MLPLinear']
    # plt.bar(xs, values)#, tick_label=labels)
    # plt.xticks(xs, labels)
    fig.tight_layout()
    plt.savefig('dsc_funeral_barchart.png', dpi=300, bbox_inches='tight')
    plt.show()

def load_and_plot_roc(name, dirs, labels):
    colors = ['orange', 'green', 'blue', 'red']
    assert len(dirs) <= 4
    for (dir, label, color) in zip(dirs, labels, colors):
        pred, target = np.load(dir)
        fpr, tpr, _ = roc_curve(target, pred)
        auc_val = auc(fpr, tpr)
        plt.plot(fpr, tpr, linestyle='--', color=color, label=f'{label} PSI events (AUC={auc_val:0.2f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.title('ROC curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive rate')

    plt.legend(loc='best')
    plt.savefig(f'{name}.png', dpi=300, bbox_inches='tight')

if __name__ == '__main__':
    bar_charts()

    # # config for HEXEvent
    # labels = ['DSC (baseline)', 'DSC (ours)', 'BiLSTM', 'D2V', 'BiLSTM + Attn']
    # experiments = ['HEXEvent_DSC', 'HEXEvent_BiLSTM', 'HEXEvent_D2V_MLP', 'HEXEvent_Attn']
    # file_name = 'pred_and_target_all.npy'
    # run_id = 'final'
    # dirs = [f'../saved/original/{file_name}']
    # for exp in experiments: dirs.append(f'../saved/log/{exp}/{run_id}/{file_name}')
    #
    # load_and_plot_roc('baseline_five_models', dirs, labels)

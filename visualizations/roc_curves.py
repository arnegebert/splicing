import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import numpy as np

# options for integrating this into my code base
# 1) save fpr / tpr from val_rounds from each run and call this afterwards with manual parameters
# 2) keep fpr / tpr in memory during an epoch and create this plot every epoch (no inter-model compar)
# 3) extract relevant bits of my code into collab where I can just call the variables

# => doing 1) and 2) at the same time probably best

def plot_and_save_roc(save_dir, *args):
    plt.cla()
    plt.style.use('seaborn')
    colors = ['orange', 'green', 'blue']
    for (((pred, target), label), color) in zip(args, colors):
        fpr, tpr, _ = roc_curve(target, pred)
        auc_val = auc(fpr, tpr)
        plt.plot(fpr, tpr, linestyle='--', color=color, label=f'{label} PSI events (AUC={auc_val:0.2f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.title('ROC curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive rate')

    plt.legend(loc='best')
    plt.savefig(f'{save_dir}/ROC.png', dpi=300, bbox_inches='tight')
    # plt.show()

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

def plot_attention(attn_weights):
    attn_weights = attn_weights[0, 0]
    plt.plot(attn_weights)
    plt.show()

if __name__ == '__main__':
    labels = ['DSC (baseline)', 'DSC (ours)', 'BiLSTM', 'D2V']

    experiments = ['HEXEvent_DSC', 'HEXEvent_BiLSTM', 'HEXEvent_D2V_MLP']
    file_name = 'pred_and_target_all.npy'
    run_id = '_final'
    dirs = [f'../saved/original/{file_name}']
    for exp in experiments: dirs.append(f'../saved/log/{exp}/{run_id}/{file_name}')

    load_and_plot_roc('baseline_four_models', dirs, labels)
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc


# options for integrating this into my code base
# 1) save fpr / tpr from val_rounds from each run and call this afterwards with manual parameters
# 2) keep fpr / tpr in memory during an epoch and create this plot every epoch (no inter-model compar)
# 3) extract relevant bits of my code into collab where I can just call the variables

# => doing 1) and 2) at the same time probably best
# =>

# plot roc curves
# plt.plot(fpr1, tpr1, linestyle='--',color='orange', label='Logistic Regression')
# plt.plot(fpr2, tpr2, linestyle='--',color='green', label='KNN')
# plt.plot(p_fpr, p_tpr, linestyle='--', color='blue')
#
# plt.title('ROC curve')
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive rate')
#
# plt.legend(loc='best')
# plt.savefig('ROC',dpi=300, bbox_inches='tight')
# plt.show()

def plot_and_save_roc(save_dir, *args):
    plt.cla()
    plt.style.use('seaborn')
    colors = ['orange', 'green', 'blue']
    for ((pred, target, label), color) in zip(args, colors):
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
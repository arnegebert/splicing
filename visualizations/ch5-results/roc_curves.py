import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import numpy as np
from matplotlib.pyplot import *

def load_and_plot_roc(name, dirs, labels):
    colors = ['orange', 'green', 'blue', 'red', 'gray']
    plt.cla()
    plt.style.use('seaborn')

    assert len(dirs) == len(labels) <= len(colors)
    for (dir, label, color) in zip(dirs, labels, colors):
        pred_and_target = np.load(dir)
        pred, target = pred_and_target[:len(pred_and_target)//2], pred_and_target[len(pred_and_target)//2:]
        fpr, tpr, _ = roc_curve(target, pred)
        auc_val = auc(fpr, tpr)
        print(f'{label}: {auc_val}')
        plt.plot(fpr, tpr, linestyle='--', label=f'{label} (AUC={auc_val:0.2f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.title('ROC curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')

    plt.legend(loc='best')
    plt.savefig(f'{name}.png', dpi=300, bbox_inches='tight')
    plt.show()


def load_and_plot_roc_shared_fig(name, dirs1, labels1, dirs2, labels2):
    colors = ['orange', 'green', 'blue', 'red', 'gray']
    plt.cla()
    plt.style.use('seaborn')

    assert len(dirs1) == len(labels1) <= len(colors)
    assert len(dirs2) == len(labels2) <= len(colors)
    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(8, 4))
    for (dir, label, color) in zip(dirs1, labels1, colors):
        pred_and_target = np.load(dir)
        pred, target = pred_and_target[:len(pred_and_target)//2], pred_and_target[len(pred_and_target)//2:]
        fpr, tpr, _ = roc_curve(target, pred)
        auc_val = auc(fpr, tpr)
        print(f'{label}: {auc_val}')
        ax1.plot(fpr, tpr, linestyle='--', label=f'{label} (AUC={auc_val:0.2f})')

    for (dir, label, color) in zip(dirs2, labels2, colors):
        pred_and_target = np.load(dir)
        pred, target = pred_and_target[:len(pred_and_target)//2], pred_and_target[len(pred_and_target)//2:]
        fpr, tpr, _ = roc_curve(target, pred)
        auc_val = auc(fpr, tpr)
        print(f'{label}: {auc_val}')
        ax2.plot(fpr, tpr, linestyle='--', label=f'{label} (AUC={auc_val:0.2f})')

    ax1.plot([0, 1], [0, 1], 'k--', lw=2)
    ax2.plot([0, 1], [0, 1], 'k--', lw=2)
    ax1.set_title('Sensory neuron cells')
    ax2.set_title('iPSC cells')
    ax1.set_xlabel('False Positive Rate')
    ax2.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')

    ax1.legend(loc='best')
    ax2.legend(loc='best')
    plt.savefig(f'{name}.png', dpi=300, bbox_inches='tight',
                )
    plt.show(dpi=300)

if __name__ == '__main__':
    # config for HEXEvent
    labels = ['DSC (original)', 'DSC (ours)', 'D2V', 'RASC']
    experiments = ['HEXEvent_DSC', 'HEXEvent_D2V_MLP', 'HEXEvent_Attn']
    file_name = 'pred_and_target_all.npy'
    run_id = 'final'
    dirs = [f'../../saved/DSC_original/{file_name}']
    for exp in experiments: dirs.append(f'../../saved/log/{exp}/{run_id}/{file_name}')
    load_and_plot_roc('hexevent_cross_model_roc_auc_comparison', dirs, labels)

    # config for HipSci SUPPA
    labels = ['DSC', 'D2V', 'RASC']
    experiments = ['HIPSCI_SUPPA_DSC', 'HIPSCI_SUPPA_D2V_MLP', 'HIPSCI_SUPPA_Attn']
    file_name = 'pred_and_target_all.npy'
    run_id = 'final'
    dirs = [f'../../saved/log/{exp}/{run_id}/{file_name}' for exp in experiments]
    load_and_plot_roc('suppa_cross_model_roc_auc_comparison', dirs, labels)

    # # config for HipSci MAJIQ neurons
    # labels = ['DSC', 'D2V', 'RASC', 'RSC']
    # experiments = ['HIPSCI_MAJIQ_DSC', 'HIPSCI_MAJIQ_D2V_MLP', 'HIPSCI_MAJIQ_Attn', 'HIPSCI_MAJIQ_BiLSTM']
    # file_name = 'pred_and_target_all.npy'
    # run_id = 'final'
    # dirs = [f'../../saved/log/{exp}/{run_id}/{file_name}' for exp in experiments]
    # load_and_plot_roc('majiq_neuron_cross_model_roc_auc_comparison', dirs, labels)
    #
    # # config for MAJIQ iPSC
    # labels = ['DSC', 'D2V', 'RASC']
    # experiments = ['iPSC_DSC', 'iPSC_D2V_MLP', 'iPSC_Attn']
    # file_name = 'pred_and_target_all.npy'
    # run_id = 'final'
    # dirs = [f'../../saved/log/{exp}/{run_id}/{file_name}' for exp in experiments]
    # load_and_plot_roc('majiq_ipsc_cross_model_roc_auc_comparison', dirs, labels)

    # config for MAJIQ neurons + iPSC
    labels1 = ['DSC', 'D2V', 'RASC', 'RSC']
    labels2 = ['DSC', 'D2V', 'RASC']
    experiments1 = ['HIPSCI_MAJIQ_DSC', 'HIPSCI_MAJIQ_D2V_MLP', 'HIPSCI_MAJIQ_Attn', 'HIPSCI_MAJIQ_BiLSTM']
    experiments2 = ['iPSC_DSC', 'iPSC_D2V_MLP', 'iPSC_Attn']
    file_name = 'pred_and_target_all.npy'
    run_id = 'final'
    dirs1 = [f'../../saved/log/{exp}/{run_id}/{file_name}' for exp in experiments1]
    dirs2 = [f'../../saved/log/{exp}/{run_id}/{file_name}' for exp in experiments2]
    load_and_plot_roc_shared_fig('majiq_neuron_ipsc_cross_model_roc_auc_comparison', dirs1, labels1, dirs2, labels2)
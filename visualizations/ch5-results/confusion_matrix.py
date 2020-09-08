import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import itertools
from sklearn.metrics import classification_report, precision_recall_curve, confusion_matrix


def plot_confusion_matrix(cm, target_names, title='', cmap=None, normalize=True, save_name=''):
    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure()#figsize=(8, 6)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    if title:
        plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=0)
        plt.yticks(tick_marks, target_names, rotation=0)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:.2f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, f"{cm[i, j]*100:.2f}%",
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
               # '\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass)
    if save_name:
        plt.savefig(save_name, dpi=300, bbox_inches='tight')
    plt.show(dpi=300)

tp, fp, fn, tn = 688, 551, 237, 2997
cf_matrix = np.array([688, 551, 237, 2997]).reshape(2,2)

# plot_confusion_matrix(cm=cf_matrix/np.sum(cf_matrix), normalize=False, target_names =['non-cons', 'cons'],
#                       title="", cmap=plt.get_cmap('rocket_r'), save_name='confusion_matrix.png')


# Outputting all stats
dir = '../../saved/log/GTEx_Junc_Brain_Attn/junc_cv_final/'


# dir = '../../saved/log/HIPSCI_MAJIQ_Attn/final/'
file = 'pred_and_target_all.npy'
pred_n_target = np.load(f'{dir}/{file}')
pred, target = pred_n_target[:len(pred_n_target)//2], pred_n_target[len(pred_n_target)//2:]

precision, recall, thresholds = precision_recall_curve(target, pred)
# convert to f score
fscore = (2 * precision * recall) / (precision + recall)
# locate the index of the largest f score
ix = np.nanargmax(fscore)
pred = pred >= thresholds[ix]
report = classification_report(target, pred, target_names=['alt', 'cons'])
print(report)
print(f'F-Score: {fscore[ix]:.2f}, Corresponding precision: {precision[ix]:.3f}, recall: {recall[ix]:.3f}')
# Note that in binary classification, recall of the positive class is also known as “sensitivity”; recall of the negative class is “specificity”

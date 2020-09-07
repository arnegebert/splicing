import numpy as np
import matplotlib.pyplot as plt

def get_values_from_log(dir_name='../../saved/log/WithConv/final', metric='loss'):
    file_name = 'info.log'
    trgt = f'{dir_name}/{file_name}'
    metric_values, metric_name_values = [], []

    end_run_msg = 'Validation performance didn\'t improve for 15 epochs. Training stops.'
    with open(trgt) as f:
        for i, line in enumerate(f):
            reports_metric = f'{metric}' in line and not f'_{metric}' in line
            reports_val_metric = f'val_{metric}' in line
            if end_run_msg in line:
                break
            if not reports_metric and not reports_val_metric: continue
            line = line.replace('\n', '')
            # removing everything until metric
            if reports_metric:
                ix = line.index(metric)
                line = line[ix:]
            elif reports_val_metric:
                ix = line.index(f'val_{metric}')
                line = line[ix:]
            # remove everything until value is reported
            ix = line.index(':')
            # +1 to remove colon itself
            line = line[ix+1:]
            value = float(line)
            if reports_metric: metric_values.append(value)
            elif reports_val_metric: metric_name_values.append(value)

    return metric_values, metric_name_values

plt.style.use('seaborn')

fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)

"""
ATTENTION
COMMENTED OUT THE FIRST TWO RUNS FOR BOTH OF THESE DIRECTORIES 
(As it's uncharacteristically poor for Baseline and few epochs for Baseline with conv leading to
a different x-axis scaling
"""
dir_heads = '../../saved/log/MultiHeads/final2'
dir_conv = '../../saved/log/MultiHeadsConv/final2'


auc, test_auc = get_values_from_log(dir_heads, 'auc')
ax1.plot(auc, label='train')
ax1.plot(test_auc, label='validation')

auc, test_auc = get_values_from_log(dir_conv, 'auc')
ax2.plot(auc, label='train')
ax2.plot(test_auc, label='validation')
# xticks_2 = np.linspace(0, len(auc), len(auc)//20)
# ax2.set_xticks(xticks_2, [0, 20, 40])



ax1.set_ylabel('AUC')
ax1.set_xlabel('epochs')
ax2.set_xlabel('epochs')

ax1.legend(loc='upper left')
# ax2.legend()

# ax1.set_xlim(0)
# ax2.set_xlim(0)
ax1.set_title('Baseline')
ax2.set_title('Baseline with additional convolution')
# even have to save this picture by copying and pasting itionto Pinta ....

plt.tight_layout()
plt.savefig('training_curve_conv_heads2.png', dpi=300, bbox='tight')

plt.show(dpi=300)

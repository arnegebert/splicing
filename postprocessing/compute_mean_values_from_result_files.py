from collections import defaultdict
import numpy as np

results_file = f'../saved/final/results_all.tsv'

targets = ['iPSC_DSC' , 'iPSC_D2V_MLP', 'iPSC_Attn']
target_values = defaultdict(list)

# collecting values
with open(results_file) as f:
    for i, line in enumerate(f):
        if not line: continue
        line = line.replace('\n', '').split('\t')
        if i == 0:
            cols = line
        else:
            name = line[0]
            if name in targets:
                values = [float(v) for v in line[1:]]
                target_values[name].append(values)

# computing averages
for name, values in target_values.items():
    print('-'*40)
    print(f'Values for experiment {name}')
    values = np.concatenate(values).reshape(len(cols)-1, -1)
    for i, col in enumerate(cols[1:]):
        print(f'{col}: {np.mean(values[:, i]):.3f} +- {np.std(values[:, i]):.3f}')

print('-'*40)
print('Done')
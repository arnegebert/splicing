import numpy as np
import matplotlib.pyplot as plt

psis = []
with open('../data/brain_cortex_full.csv') as f:
    for l in f:
        j, start_seq, end_seq, psi = l.split(',')
        psis.append(float(psi[:-1]))

psis = np.array(psis)
length = len(psis)
print(f'number of values: {len(psis)}')
print(f'percent of zeroes: {np.sum(psis==0)/length}')
print(f'percent of psis <0.5: {np.sum(psis<0)/length}')
print(f'percent of psis =0.5: {np.sum(psis==0.5)/length}')
print(f'percent of psis >0.5: {np.sum(psis>0.5)/length}')
print(f'percent of psis =1: {np.sum(psis==1)/length}')
print(f'mean: {np.mean(psis)}')
print(f'median: {np.mean(psis)}')

plt.hist(psis)
plt.xlabel('PSI value')
plt.ylabel('number of data points')
plt.show()

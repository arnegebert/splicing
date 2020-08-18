import numpy as np
import matplotlib.pyplot as plt

# will plot hexevent / gtx / majiq / suppa as most variation + 4 fit better into plot
from matplotlib import ticker

src_paths = ['../data/hexevent', '../data/gtex_processed/brain', '../data/hipsci_suppa',
             '../data/hipsci_majiq/exon',]
names = ['HEXEvent', 'GTEx (exon)', 'HipSci SUPPA (exon)', 'HipSci MAJIQ (exon)']

# https://matplotlib.org/3.1.0/gallery/subplots_axes_and_figures/subplots_demo.html
fig, axs = plt.subplots(2, 2)
for i, (src_path, name) in enumerate(zip(src_paths, names)):
    cons = np.load(f'{src_path}/cons.npy')
    low = np.load(f'{src_path}/low.npy')
    high = np.load(f'{src_path}/high.npy')
    psis = np.concatenate((cons[:, -1, 3], low[:, -1, 3], high[:, -1, 3]), axis=0)
    ax0, ax1 = 1 if i >= 2 else 0, i % 2
    axs[ax0, ax1].hist(psis)
    axs[ax0, ax1].set_title(name)
    # https://malithjayaweera.com/2018/09/add-matplotlib-percentage-ticks-histogram/
    axs[ax0, ax1].yaxis.set_major_formatter(ticker.PercentFormatter(xmax=len(psis)))

# want to have x-axis display relative number of data points and should be the same across all datasets
for ax in axs.flat:
    ax.set(xlabel='PSI value', ylabel='number of data points')

# Hide x labels and tick labels for top plots and y ticks for right plots.
for ax in axs.flat:
    ax.label_outer()

# todo; make all axes scale in the same way
# with open('../data/gtex_origin/brain_cortex_junction_seqs.csv') as f:
#     for l in f:
#         # j, start_seq, end_seq, psi = l.split(',')
#         j, start_seq, end_seq, psi = l.split('\t')
#         psis.append(float(psi[:-1]))
#
# psis = np.array(psis)
# length = len(psis)
# print(f'number of values: {length}')
# print(f'percent of zeroes: {np.sum(psis==0)/length}')
# print(f'percent of psis <0.5: {np.sum(psis<0.5)/length}')
# print(f'percent of psis =0.5: {np.sum(psis==0.5)/length}')
# print(f'percent of psis >0.5: {np.sum(psis>0.5)/length}')
# print(f'percent of psis =1: {np.sum(psis==1)/length}')
# print(f'mean: {np.mean(psis)}')
# print(f'median: {np.median(psis)}')
#
# plt.hist(psis)
# plt.xlabel('PSI value')
# plt.ylabel('number of data points')
plt.show()

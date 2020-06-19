import numpy as np
import matplotlib.pyplot as plt

totals = [0] * 255
with open(f'../data/brain_cortex_junction_reads.csv') as f:
    for l in f:
        l = l.split(',')
        reads = l[1].split('\t')
        if len(reads) == 1: continue
        for i, r in enumerate(reads):
            totals[i] += int(r)


totals = np.array(totals)
print(f'Sample index with most total reads ({np.max(totals)}): {np.argmax(totals)}')
plt.plot(np.arange(255), totals)
plt.xlabel('sample index')
plt.ylabel('number of reads')

print(f'Mean read count: {np.mean(totals)}')
print(f'Median read count: {np.median(totals)}')
print(f'Min read count: {np.min(totals)}')

plt.show()
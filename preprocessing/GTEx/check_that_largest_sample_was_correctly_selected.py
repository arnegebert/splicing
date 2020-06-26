import numpy as np

total = 0
with open(f'../../data/gtex_processed/brain_cortex_junction_reads_one_sample.csv') as f:
    for i, line in enumerate(f):
        if i % 1000 == 0: print(f'Processing line {i}')
        line = line.split(',')
        total += int(line[1])

# Total number of reads: 30694056
print(f'Total number of reads: {total}')
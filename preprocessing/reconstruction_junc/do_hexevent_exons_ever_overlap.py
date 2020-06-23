"""Answer: yes"""

import csv
src = '../../data/dsc_reconstruction_junction/cons_exons.csv'

with open(src) as f:
    r = csv.reader(f, delimiter='\t')
    cass = list(r)


def overlap(start, end, start2, end2):
    return not (start > end2 or end < start2)

prev_start, prev_end = 0, 0
overlaps = 0
for i, (chrom, strand, start, end, count, skip, constit_level) in enumerate(cass):
    if i % 1000 == 0: print(f'Reading line {i}')
    start, end = int(start), int(end)
    if overlap(prev_start, prev_end, start, end):
        overlaps += 1
        print(f'{chrom}: {strand}')
        print(f'{prev_start}:{prev_end} vs. {start}:{end}')
        print(constit_level)
    prev_start, prev_end = start, end

print(overlaps)
print('Finished')
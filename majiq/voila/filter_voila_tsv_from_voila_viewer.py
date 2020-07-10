import csv

# idea: build set of LSV ids given
lsvs = set()
with open('binary_exon_skipping_junc_all.txt') as f:
    for line in f:
        lsvs.add(line[:-1])

filtered = []
with open('all.tsv') as f:
    for i, line in enumerate(f):
        if i % 1000 == 0: print(f'Line {i}')
        line = line.split('\t')
        if len(line) < 10: continue
        lsvid = line[2]
        if lsvid in lsvs or lsvid == 'lsv_id':
            filtered.append('\t'.join(line))

print(f'Number of LSVs after filtering: {len(filtered)}')
with open('filtered_all.tsv', 'w') as f:
    for l in filtered:
        f.write(l)
from collections import defaultdict

def convert(str_bool):
    return str_bool == 'True'

es_lines = []
d = defaultdict(lambda: 0)

with open('all.psi.tsv') as f:
    for i, line in enumerate(f):
        if i == 0: continue
        geneid, lsvid, lsvtype, epsi, stdev, a5ss, a3ss, es, njunc, nexon, juncoord, ircoord = line.split('\t')
        a5ss, a3ss, es = convert(a5ss), convert(a3ss), convert(es)
        if es and not a5ss and not a3ss:
            es_lines.append(line)
            d[njunc] += 1

print(f'Number of exon skipped events: {len(es_lines)}') # 26641
for (k, v) in d.items():
    print(f'{k}: {v}')


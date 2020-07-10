from collections import defaultdict

def convert(str_bool):
    return str_bool == 'True'

es_lines = []
exons = []
var1, var2 = 0, 0
d = defaultdict(lambda: 0)
prev_start1 = 0
with open('all.psi.tsv') as f:
    for i, line in enumerate(f):
        if i == 0: continue
        geneid, lsvid, lsvtype, epsi, stdev, a5ss, a3ss, es, njunc, nexon, juncoord, ircoord = line.replace('\n','').split('\t')
        a5ss, a3ss, es = convert(a5ss), convert(a3ss), convert(es)
        if es and not a5ss and not a3ss:
            if njunc == '2':
                idx = juncoord.find(';')
                coord1, coord2 = juncoord[:idx], juncoord[idx+1:]
                if ';' in coord2: continue
                idx1, idx2 = coord1.find('-'), coord2.find('-')
                start1, end1 = int(coord1[:idx1]), int(coord1[idx1 + 1:])
                start2, end2 = int(coord2[:idx2]), int(coord2[idx2 + 1:])
                if start1 == start2: var1 +=1
                elif end1 == end2: var2 += 1

                es_lines.append((geneid, lsvid, lsvtype, epsi, stdev, a5ss, a3ss, es, njunc, nexon, juncoord, ircoord,
                                 start1, end1, start2, end2))
                if start1 == prev_start1:
                    exons.append(es_lines[-1])
                prev_start1 = start1

            d[njunc] += 1


es_lines.sort(key=lambda tup: tup[-1])
es_lines.sort(key=lambda tup: tup[-2])
es_lines.sort(key=lambda tup: tup[-3])
es_lines.sort(key=lambda tup: tup[-4])

print(f'No var1: {var1}')
print(f'No var2: {var2}')
print(f'Number of legitimate cassette exons: {len(exons)}')

print(f'Number of exon skipped events: {len(es_lines)}') # 26641
for (k, v) in d.items():
    print(f'{k}: {v}')

with open('sorted_all.psi.tsv', 'w') as f:
    f.write(f'geneid\tlsvid\tlsvtype\tE[psi]\tStd[E[psi]]\ta5ss\ta3ss\tES\tnjunc\tnexon\tjunc coord\tir coord\tstart1\tend1\tstart2\tend2\n')
    for (geneid, lsvid, lsvtype, epsi, stdev, a5ss, a3ss, es, njunc, nexon, juncoord, ircoord,
                                 start1, end1, start2, end2) in es_lines:
        f.write(f'{geneid}\t{lsvid}\t{lsvtype}\t{epsi}\t{stdev}\t{a5ss}\t{a3ss}\t{es}\t{njunc}\t{nexon}\t{juncoord}\t{ircoord}\t{start1}\t{end1}\t{start2}\t{end2}\n')
        
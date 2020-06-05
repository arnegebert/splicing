

exons = []
with open('../../data/gencode.v34.annotation.gtf') as f:
    for i, l in enumerate(f):
        if i >= 5:
            l = l.split('\t')
            if len(l) == 1: continue
            type = l[2]
            if type != 'exon': continue
            chr, start, end = l[0], int(l[3]), int(l[4])
            start = start - 1
            exons.append((chr, start, end))

# sort by end then start to have list sorted by start and sub-ordered by end
exons.sort(key=lambda tup: tup[2])
exons.sort(key=lambda tup: tup[1])

with open('../../data/gencode_exon.csv', 'w') as f:
    for (chr, start, end) in exons:
        f.write(f'{chr}\t{start}\t{end}\n')
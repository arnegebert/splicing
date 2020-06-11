

exons = []
with open('../../data/gencode.v34.annotation.gtf') as f:
    for i, l in enumerate(f):
        if i >= 5:
            l = l.split('\t')
            if len(l) == 1: continue
            type = l[2]
            if type != 'gene': continue
            chr, start, end = l[0], int(l[3]), int(l[4])
            freetext = l[8]
            gene_id_idx = freetext.index(';')
            gene_id = freetext[9:gene_id_idx-1]
            try:
                chr_number = int(l[0][3:])
            except ValueError:
                continue
            start = start - 1
            exons.append((gene_id, chr, start, end))

# sort by end then start to have list sorted by start and sub-ordered by end
exons.sort(key=lambda tup: tup[3])
exons.sort(key=lambda tup: tup[2])
exons.sort(key=lambda tup: tup[1])

print('Writing filtered data')
with open('../../data/gencode_genes.csv', 'w') as f:
    for (gene_id, chr, start, end) in exons:
        f.write(f'{gene_id}\t{chr}\t{start}\t{end}\n')


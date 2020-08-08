import csv

indv_name = 'all2'
# idea: build set of LSV ids given
lsvs = set()
with open(f'binary_exon_skipping_junc_{indv_name}.txt') as f:
    for line in f:
        lsvs.add(line[:-1])

filtered = []
with open(f'{indv_name}.tsv') as f:
    for i, line in enumerate(f):
        if i % 1000 == 0: print(f'Line {i}')
        line = line.split('\t')
        if len(line) < 10: continue
        genename, geneid, lsvid, epsi, stdev, lsvtype, njunc, nexon, denovojunc, \
        chrom, strand, juncoord, excoords, ircoord, ucsc = line
        if lsvid in lsvs:
            try:
                chrom = int(chrom[3:])
            except ValueError:
                continue
            coord1, coord2 = juncoord.split(';')
            idx1, idx2 = coord1.find('-'), coord2.find('-')
            s1, e1 = int(coord1[:idx1]), int(coord1[idx1 + 1:])
            s2, e2 = int(coord2[:idx2]), int(coord2[idx2 + 1:])
            filtered.append((genename, geneid, lsvid, epsi, stdev, lsvtype, njunc, nexon, denovojunc,
        chrom, strand, juncoord, excoords, ircoord, ucsc, s1, e1, s2, e2))

filtered.sort(key=lambda tup: tup[-1])
filtered.sort(key=lambda tup: tup[-2])
filtered.sort(key=lambda tup: tup[-3])
filtered.sort(key=lambda tup: tup[-4])
filtered.sort(key=lambda tup: tup[9])

print(f'Number of LSVs after filtering: {len(filtered)}')
with open(f'filtered_{indv_name}.tsv', 'w') as f:
    header = ['gene_name', 'gene_id', 'lsv_id', 'mean_psi_per_lsv_junction', 'stdev_psi_per_lsv_junction', 'lsv_type', 'num_junctions', 'num_exons', 'de_novo_junctions', 'seqid', 'strand', 'junctions_coords', 'exons_coords', 'ir_coords', 'ucsc_lsv_link\n']
    f.write('\t'.join(header))
    for (genename, geneid, lsvid, epsi, stdev, lsvtype, njunc, nexon, denovojunc,
        chrom, strand, juncoord, excoords, ircoord, ucsc, s1, e1, s2, e2) in filtered:
        f.write(f'{genename}\t{geneid}\t{lsvid}\t{epsi}\t{stdev}\t{lsvtype}\t{njunc}\t{nexon}\t{denovojunc}\t{chrom}\t{strand}\t{juncoord}\t{excoords}\t{ircoord}\t{ucsc}')
from utils import timer

juncs = []
nproblematic_jstart = 0
nproblematic_jend = 0
with open('constitutive_junctions.tsv') as f:
    for i, line in enumerate(f):
        if i==0: continue
        geneid, chrom, jstart, jend, dstart, dend, astart, aend = line.replace('\n','').split('\t')
        try:
            chrom = int(chrom[3:])
        except ValueError:
            continue
        jstart, jend, dstart, dend, astart, aend = int(jstart), int(jend), int(dstart), int(dend),\
                                                            int(astart), int(aend)
        if jstart != dend:
            nproblematic_jstart += 1
            continue
        if jend != astart:
            nproblematic_jend += 1
            continue
        juncs.append((geneid, chrom, jstart, jend, dstart, dend, astart, aend))

for i in range(7, 1-1, -1):
    juncs.sort(key=lambda tup: tup[i])

prev_junc = None
ndupl = 0
with open('constitutive_junctions_sorted.tsv', 'w') as f:
    f.write(f'#GENEID\tCHROMOSOME\tJUNC_START\tJUNC_END\tDONOR_START\tDONOR_END\tACCEPTOR_START\tACCEPTOR_END\n')
    for (geneid, chrom, jstart, jend, dstart, dend, astart, aend) in juncs:
        if (jstart, jend) == prev_junc:
            ndupl += 1
            continue
        f.write(f'{geneid}\t{chrom}\t{jstart}\t{jend}\t{dstart}\t{dend}\t{astart}\t{aend}\n')
        prev_junc = (jstart, jend)

print(f'Number sorted, filtered constitutive junctions: {len(juncs)}')
# these are weird and I don't know why this is happening
print(f'Number of junctions where jstart != dend: {nproblematic_jstart}')
print(f'Number of junctions where jend != astart: {nproblematic_jend}')

print(f'Number of duplicate junctions: {ndupl}')


@timer
def conv_junc_to_exon():
    """ Converting sorted junctions to exons here """
    def overlap(start, end, start2, end2):
        return not (start > end2 or end < start2)

    cons_exons = []
    overlaps = 0
    prev_astart, prev_aend = juncs[0][-2], juncs[0][-1]
    prev_jstart, prev_jend = juncs[0][2], juncs[0][3]
    for i, (geneid, chrom, jstart, jend, dstart, dend, astart, aend) in enumerate(juncs, 1):
        if (prev_astart, prev_aend) == (dstart, dend):
            cons_exons.append((geneid, chrom, dstart, dend))
        if overlap(prev_jstart, prev_jend, jstart, jend):
            overlaps += 1
        # if prev_aend == dend:
        #     cons_exons.append((geneid, chrom, jstart, jend, dstart, dend, astart, aend))
        prev_astart, prev_aend = astart, aend
        prev_jstart, prev_jend = jstart, jend

    print(f'Number of constitutive exons found this way: {len(cons_exons)}')
    print(f'Number of overlapping (supposedly constitutive) junctions: {overlaps}')

    with open('constitutive_exons.tsv', 'w') as f:
        for (geneid, chrom, dstart, dend) in cons_exons:
            f.write(f'{geneid}\t{chrom}\t{dstart}\t{dend}\n')
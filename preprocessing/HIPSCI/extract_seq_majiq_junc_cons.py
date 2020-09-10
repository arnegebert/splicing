import numpy as np
import time
from utils import one_hot_encode_seq, reverse_complement

print(
"""
DEPRECATED - CHECK THAT EXON BOUNDARIES ARE CORRECT FOR NUCLEOTIDE EXTRACTION BEFORE USING
"""
)

startt = time.time()
psis = []

data_path = '../../data'
save_to_cons = 'hipsci_majiq/junc/cons.npy'
introns_bef_start = 70 # introns
exons_after_start = 70 # exons
exons_bef_end = 70 # exons
introns_after_end = 70 # introns

# want to load chromosome as one giant string
def load_chrom_seq(chrom):
    with open(f'../../data/chromosomes/chr{chrom}.fa') as f:
        loaded_chrom_seq = f.read().replace('\n', '')
        if chrom < 10:
            return loaded_chrom_seq[5:]
        else:
            return loaded_chrom_seq[6:]

cons_juncs = []
exon_mean, exon_std, intron_mean, intron_std = 145.42, 198.0, 5340., 17000.
with open('../../majiq/builder/constitutive_junctions_sorted_stranded.tsv') as f:
    loaded_chrom = 1
    chrom_seq = load_chrom_seq(loaded_chrom)
    for i, l in enumerate(f):
        if i==0: continue
        if i % 1000 == 0: # ~ 357500 junctions
            print(f'Reading line {i}')

        geneid, chrom, jstart, jend, dstart, dend, astart, aend, strand = l.replace('\n','').split('\t')
        chrom, jstart, jend, dstart, aend, psi = int(chrom), int(jstart), int(jend), int(dstart), int(aend), 1.0
        if chrom > loaded_chrom:
            loaded_chrom += 1
            chrom_seq = load_chrom_seq(loaded_chrom)

        window_around_start = chrom_seq[jstart-introns_bef_start-1:jstart+exons_after_start-1]
        window_around_end = chrom_seq[jend-exons_bef_end-2:jend+introns_after_end-2]
        if strand == '-':
            window_around_start, window_around_end = reverse_complement(window_around_end[::-1]), \
                                                     reverse_complement(window_around_start[::-1])

        start, end = one_hot_encode_seq(window_around_start), one_hot_encode_seq(window_around_end)
        start, end = np.array(start), np.array(end)
        l1, l2, l3 = jstart-dstart, jend-jstart, aend-jend
        l1, l2, l3 = (l1-exon_mean)/exon_std, (l2-intron_mean)/intron_std, (l3-exon_mean)/exon_std
        lens_and_psi_vector = np.array([l1, l2, l3, psi])
        start_and_end = np.concatenate((start, end))
        sample = np.concatenate((start_and_end,lens_and_psi_vector.reshape(1,4))).astype(np.float32)
        cons_juncs.append(sample)

        psis.append(psi)
    cons_juncs = np.array(cons_juncs)

psis = np.array(psis)

print(f'Number of cons junctions: {len(cons_juncs)}')

np.save(f'{data_path}/{save_to_cons}', cons_juncs)
print(f'Runtime {time.time()-startt}')
# with vs without formatting doesn't make a difference
# 18360
# 0.4743559224421955
# 0.2023384176178008
# sum(events==1) = 7900
# sum(events==0) = 8792
# sum(events>=0.99) = 7955
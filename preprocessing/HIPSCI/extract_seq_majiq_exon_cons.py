import numpy as np
import time
from utils import one_hot_encode_seq, reverse_complement, overlap
import matplotlib.pyplot as plt

startt = time.time()
psis = []

data_path = '../../data'
save_to_cons = 'hipsci_majiq/exon/cons.npy'
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

l1s, l2s, l3s = [], [], []
cons_exons = []
prev_astart, prev_aend = 0, 0
prev_dstart, prev_dend = 0, 0
prev_jstart, prev_jend = 0, 0
overlaps = 0
exon_mean, exon_std, intron_mean, intron_std = 145.42, 198.0, 5340., 17000.
with open('../../majiq/builder/constitutive_junctions_sorted_stranded.tsv') as f:
    loaded_chrom = 1
    chrom_seq = load_chrom_seq(loaded_chrom)
    for i, l in enumerate(f):
        if i==0: continue
        if i % 1000 == 0: # ~ 357500 junctions
            print(f'Reading line {i}')

        geneid, chrom, jstart, jend, dstart, dend, astart, aend, strand = l.replace('\n','').split('\t')
        chrom, jstart, jend, dstart, dend, astart, aend, psi = int(chrom), int(jstart), int(jend), int(dstart), \
                                                               int(dend), int(astart), int(aend), 1.0
        if chrom > loaded_chrom:
            loaded_chrom += 1
            chrom_seq = load_chrom_seq(loaded_chrom)

        if overlap(prev_jstart, prev_jend, jstart, jend):
            overlaps += 1
        if (prev_astart, prev_aend) == (dstart, dend):

            window_around_start = chrom_seq[dstart-introns_bef_start-1:dstart+exons_after_start-1]
            window_around_end = chrom_seq[dend-exons_bef_end-2:dend+introns_after_end-2]
            if strand == '-':
                window_around_start, window_around_end = reverse_complement(window_around_end[::-1]), \
                                                         reverse_complement(window_around_start[::-1])

            start, end = one_hot_encode_seq(window_around_start), one_hot_encode_seq(window_around_end)
            start, end = np.array(start), np.array(end)
            l1, l2, l3 = dstart-prev_dend, dend-dstart, astart-dend
            l1, l2, l3 = (l1-intron_mean)/intron_std, (l2-exon_mean)/exon_std, (l3-intron_mean)/intron_std
            lens_and_psi_vector = np.array([l1, l2, l3, psi])
            l1s.append(l1)
            l2s.append(l2)
            l3s.append(l3)
            start_and_end = np.concatenate((start, end))
            sample = np.concatenate((start_and_end,lens_and_psi_vector.reshape(1,4))).astype(np.float32)
            cons_exons.append(sample)

            psis.append(psi)
        prev_astart, prev_aend = astart, aend
        prev_dstart, prev_dend = dstart, dend
        prev_jstart, prev_jend = jstart, jend
    cons_exons = np.array(cons_exons)

psis = np.array(psis)

l1s, l2s, l3s = np.array(l1s), np.array(l2s), np.array(l3s)
l1avg, l2avg, l3avg = np.mean(l1s), np.mean(l2s), np.mean(l3s)
l1median, l2median, l3median = np.median(l1s), np.median(l2s), np.median(l3s)

plt.hist(l1s)
plt.xlabel('normalized L1 value')
plt.ylabel('number of data points')
plt.title('Constitutive exons MAJIQ')
plt.show()

print(f'L1avg: {l1avg}, l2avg: {l2avg}, l3avg: {l3avg}')
print(f'L1 median: {l1median}, l2 median: {l2median}, l3 median: {l3median}')


print(f'Number of cons exon: {len(cons_exons)}')
print(f'Number of overlapping (supposedly constitutive) junctions: {overlaps}')

np.save(f'{data_path}/{save_to_cons}', cons_exons)
print(f'Runtime {time.time()-startt}')
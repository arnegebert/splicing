import numpy as np
import time
from utils import one_hot_encode_seq, reverse_complement

startt = time.time()
psis = []
introns_bef_start = 70 # introns
exons_after_start = 70 # exons
data_path = '../../data'
save_to_low = 'hipsci_suppa/low.npy'
save_to_high = 'hipsci_suppa/high.npy'
save_to_cons = 'hipsci_suppa/cons.npy'
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

cons_exons, high_exons, low_exons = [], [], []
exon_mean, exon_std, intron_mean, intron_std = 145.42, 198.0, 5340., 17000.
with open('../../suppa/formatted_second.psi') as f:
    loaded_chrom = 1
    chrom_seq = load_chrom_seq(loaded_chrom)
    for i, l in enumerate(f):
        if i==0: continue
        if i % 1000 == 0: # ~ 357500 junctions
            print(f'Reading line {i}')

        chrom, strand, e1, s1, e2, s2, psi = l.replace('\n','').split('\t')
        chrom, e1, s1, e2, s2, psi = int(chrom), int(e1), int(s1), int(e2), int(s2), float(psi)
        if chrom > loaded_chrom:
            loaded_chrom += 1
            chrom_seq = load_chrom_seq(loaded_chrom)

        window_around_start = chrom_seq[s1-introns_bef_start-1:s1+exons_after_start-1]
        window_around_end = chrom_seq[e2-exons_bef_end-2:e2+introns_after_end-2]
        if strand == '-':
            window_around_start, window_around_end = reverse_complement(window_around_end[::-1]), \
                                                     reverse_complement(window_around_start[::-1])
            # window_around_start, window_around_end = reverse_complement(window_around_start[::-1]), \
            #                                          reverse_complement(window_around_end[::-1])
            # window_around_start, window_around_end = reverse_complement(window_around_start), \
            #                                          reverse_complement(window_around_end)

        # not doing anything based on strand type kinda tends to have the best results....

        start, end = one_hot_encode_seq(window_around_start), one_hot_encode_seq(window_around_end)
        start, end = np.array(start), np.array(end)
        l1, l2, l3 = s1-e1, e2-s1, s2-e2
        l1, l2, l3 = (l1-intron_mean)/intron_std, (l2-exon_mean)/exon_std, (l3-intron_mean)/intron_std
        lens_and_psi_vector = np.array([l1, l2, l3, psi])
        start_and_end = np.concatenate((start, end))
        sample = np.concatenate((start_and_end,lens_and_psi_vector.reshape(1,4))).astype(np.float32)
        if psi < 0.8:
            low_exons.append(sample)
        elif psi < 1:
            high_exons.append(sample)
        else:
            cons_exons.append(sample)

        psis.append(psi)
    low_psi_exons = np.array(low_exons)
    high_psi_exons = np.array(high_exons)
    cons_exons = np.array(cons_exons)

psis = np.array(psis)
print(f'Number of samples: {len(psis)}')
print(f'Mean PSI: {np.mean(psis)}')
print(f'Median PSI: {np.median(psis)}')

print(f'Number of low PSI exons: {len(low_exons)}')
print(f'Number of high PSI exons: {len(high_exons)}')
print(f'Number of cons exons: {len(cons_exons)}')

np.save(f'{data_path}/{save_to_low}', low_exons)
np.save(f'{data_path}/{save_to_high}', high_exons)
np.save(f'{data_path}/{save_to_cons}', cons_exons)
print(f'Runtime {time.time()-startt}')
# with vs without formatting doesn't make a difference
# 18360
# 0.4743559224421955
# 0.2023384176178008
# sum(events==1) = 7900
# sum(events==0) = 8792
# sum(events>=0.99) = 7955
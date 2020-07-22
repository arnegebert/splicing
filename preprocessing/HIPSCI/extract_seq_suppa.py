import numpy as np
import time
from utils import one_hot_encode_seq, reverse_complement
import matplotlib.pyplot as plt

startt = time.time()
psis = []

data_path = '../../data'
save_to_low = 'hipsci_suppa/low.npy'
save_to_high = 'hipsci_suppa/high.npy'
save_to_cons = 'hipsci_suppa/cons.npy'
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

l1scons, l2scons, l3scons = [], [], []
l1scass, l2scass, l3scass = [], [], []
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
            # print('hello....')

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
            l1scass.append(l1)
            l2scass.append(l2)
            l3scass.append(l3)
        elif psi < 1:
            high_exons.append(sample)
            l1scass.append(l1)
            l2scass.append(l2)
            l3scass.append(l3)
        else:
            cons_exons.append(sample)
            l1scons.append(l1)
            l2scons.append(l2)
            l3scons.append(l3)

        psis.append(psi)
    low_psi_exons = np.array(low_exons)
    high_psi_exons = np.array(high_exons)
    cons_exons = np.array(cons_exons)

l1scons, l2scons, l3scons = np.array(l1scons), np.array(l2scons), np.array(l3scons)
l1avgcons, l2avgcons, l3avgcons = np.mean(l1scons), np.mean(l2scons), np.mean(l3scons)
l1mediancons, l2mediancons, l3mediancons = np.median(l1scons), np.median(l2scons), np.median(l3scons)

print(f'Cons:')
print(f'L1avg: {l1avgcons}, l2avg: {l2avgcons}, l3avg: {l3avgcons}')
print(f'L1 median: {l1mediancons}, l2 median: {l2mediancons}, l3 median: {l3mediancons}')

l1scass, l2scass, l3scass = np.array(l1scass), np.array(l2scass), np.array(l3scass)
l1avgcass, l2avgcass, l3avgcass = np.mean(l1scass), np.mean(l2scass), np.mean(l3scass)
l1mediancass, l2mediancass, l3mediancass = np.median(l1scass), np.median(l2scass), np.median(l3scass)

print(f'Cass:')
print(f'L1avg: {l1avgcass}, l2avg: {l2avgcass}, l3avg: {l3avgcass}')
print(f'L1 median: {l1mediancass}, l2 median: {l2mediancass}, l3 median: {l3mediancass}')

plt.hist(l1scons)
plt.xlabel('normalized L1 value')
plt.ylabel('number of data points')
plt.title('Constitutive exons SUPPA')
plt.show()

plt.hist(l1scass)
plt.xlabel('normalized L1 value')
plt.ylabel('number of data points')
plt.title('Cassette exons SUPPA')
plt.show()

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
from collections import defaultdict
import time
import numpy as np
from utils import reverse_complement, one_hot_encode_seq, intron_mean, exon_mean, intron_std, exon_std
import matplotlib.pyplot as plt

print(
"""
DEPRECATED - CHECK THAT EXON BOUNDARIES ARE CORRECT FOR NUCLEOTIDE EXTRACTION BEFORE USING (see majiq neuron)
\n"""*10
)

startt = time.time()
# want to load chromosome as one giant string
def load_chrom_seq(chrom):
    with open(f'../../data/chromosomes/chr{chrom}.fa') as f:
        loaded_chrom_seq = f.read().replace('\n', '')
        if chrom < 10:
            return loaded_chrom_seq[5:]
        else:
            return loaded_chrom_seq[6:]

data_path = '../../data'
sample_name = 'lexy2'
save_to_low = f'iPSC/exon/low_{sample_name}.npy'
save_to_high = f'iPSC/exon/high_{sample_name}.npy'

introns_bef_start = 70 # introns
exons_after_start = 70 # exons
exons_bef_end = 70 # exons
introns_after_end = 70 # introns

l1s, l2s, l3s = [], [], []
psis = []
cons_exons, high_exons, low_exons = [], [], []
var1, var2, var3 = 0, 0, 0
shorter_first, shorter_second = 0, 0
shorter_first_var1, shorter_second_var1 = 0, 0
shorter_first_var2, shorter_second_var2 = 0, 0
junc_not_matching_exon = 0
d = defaultdict(lambda: 0)
prev_start1 = 0
denovo = 0
with open(f'../../majiq/voila/filtered_{sample_name}.tsv') as f:
    loaded_chrom = 1
    chrom_seq = load_chrom_seq(loaded_chrom)

    for i, line in enumerate(f):
        if i == 0: continue
        if i % 1000 == 0: print(f'Line {i}')
        genename, geneid, lsvid, epsi, stdev, lsvtype, njunc, nexon, denovojunc, \
        chrom, strand, juncoord, excoords, ircoord, ucsc = line.replace('\n', '').split('\t')
        denovo += denovojunc.count('1')

        coord1, coord2 = juncoord.split(';')
        idx1, idx2 = coord1.find('-'), coord2.find('-')
        s1, e1 = int(coord1[:idx1]), int(coord1[idx1 + 1:])
        s2, e2 = int(coord2[:idx2]), int(coord2[idx2 + 1:])

        excoord1, excoord2, excoord3 = excoords.split(';')
        exidx1, exidx2, exidx3 = excoord1.find('-'), excoord2.find('-'), excoord2.find('-')
        try:
            ex1start, ex1end = int(excoord1[:exidx1]), int(excoord1[exidx1 + 1:])
            ex2start, ex2end = int(excoord2[:exidx2]), int(excoord2[exidx2 + 1:])
            ex3start, ex3end = int(excoord3[:exidx3]), int(excoord3[exidx3 + 1:])
        except ValueError: continue

        if s1 == s2:
            var1 += 1
            if (e1 - s1) < (e2 - s2):
                shorter_first_var1 += 1
            else:
                shorter_second_var1 += 1
        elif e1 == e2:
            var2 += 1
            if (e1 - s1) < (e2 - s2):
                shorter_first_var2 += 1
            else:
                shorter_second_var2 += 1
        else:
            var3 += 1
        if (e1 - s1) < (e2 - s2):
            shorter_first += 1
        else: shorter_second += 1
        for coord in [s1, s2, e1, e2]:
            if str(coord) not in excoords:
                junc_not_matching_exon += 1
                break
        # copy from other extract sequence files:
        # extract sequence
        if int(chrom) > loaded_chrom:
            loaded_chrom += 1
            chrom_seq = load_chrom_seq(loaded_chrom)

        # select the psi whose junction includes the exon
        psi1, psi2 = map(float, epsi.split(';'))
        if (e1 - s1) < (e2 - s2):
            psi = psi1
        else: psi = psi2
        window_around_start = chrom_seq[ex2start - introns_bef_start - 1:ex2start + exons_after_start - 1]
        window_around_end = chrom_seq[ex2end - exons_bef_end - 2:ex2end + introns_after_end - 2]
        if strand == '-':
            window_around_start, window_around_end = reverse_complement(window_around_end[::-1]), \
                                                     reverse_complement(window_around_start[::-1])
        # # encode & convert to numpy arrays
        start, end = one_hot_encode_seq(window_around_start), one_hot_encode_seq(window_around_end)
        start, end = np.array(start), np.array(end)

        l1, l2, l3 = ex2start - ex1end, ex2end - ex2start, ex3start - ex2end
        l1, l2, l3 = (l1 - intron_mean) / intron_std, (l2 - exon_mean) / exon_std, (l3 - intron_mean) / intron_std
        l1s.append(l1)
        l2s.append(l2)
        l3s.append(l3)

        lens_and_psi_vector = np.array([l1, l2, l3, psi])
        start_and_end = np.concatenate((start, end))
        sample = np.concatenate((start_and_end,lens_and_psi_vector.reshape(1,4))).astype(np.float32)
        if psi < 0.8:
            low_exons.append(sample)
        elif psi < 1:
            high_exons.append(sample)
        else:
            raise Exception('Constitutive exon detected')
        psis.append(psi)
    low_psi_exons = np.array(low_exons)
    high_psi_exons = np.array(high_exons)

psis = np.array(psis)
l1s, l2s, l3s = np.array(l1s), np.array(l2s), np.array(l3s)
l1avg, l2avg, l3avg = np.mean(l1s), np.mean(l2s), np.mean(l3s)
l1median, l2median, l3median = np.median(l1s), np.median(l2s), np.median(l3s)

print(f'L1avg: {l1avg}, l2avg: {l2avg}, l3avg: {l3avg}')
print(f'L1 median: {l1median}, l2 median: {l2median}, l3 median: {l3median}')

plt.hist(l1s[l1s<=50])
plt.xlabel('normalized L1 value')
plt.ylabel('number of data points')
plt.title('Cassette exons MAJIQ')
# plt.show()


print(f'Number of samples: {len(psis)}')
print(f'Mean PSI: {np.mean(psis)}')
print(f'Median PSI: {np.median(psis)}')

print(f'Number of generated training samples: {len(low_psi_exons)+len(high_psi_exons)+len(cons_exons)}')  # 22700

print(f'Number of low PSI exons: {len(low_exons)}')
print(f'Number of high PSI exons: {len(high_exons)}')
print(f'Number of cons exons: {len(cons_exons)}')

np.save(f'{data_path}/{save_to_low}', low_exons)
np.save(f'{data_path}/{save_to_high}', high_exons)
print(f'Runtime {time.time()-startt}')

print(f'Number of junctions where cassette exon is left: {var1}') # 10886
print(f'Number of junctions where cassette exon is right: {var2}') # 10952
print(f'Number of junctions where cassette exon is neither left nor right: {var3}') # 862
print(f'Number of junctions where shorter junction is first: {shorter_first}') # 10899
print(f'Number of junctions where shorter junction is second: {shorter_second}') # 11801

print(f'Number of var1 junctions where shorter junction is first: {shorter_first_var1}') # 5379
print(f'Number of var1 junctions where shorter junction is second: {shorter_second_var1}') # 5507
print(f'Number of var2 junctions where shorter junction is first: {shorter_first_var2}') # 5076
print(f'Number of var2 junctions where shorter junction is second: {shorter_second_var2}') # 5876
print(f'Number of de novo junctions: {denovo}')
for (k, v) in d.items():
    print(f'{k}: {v}')
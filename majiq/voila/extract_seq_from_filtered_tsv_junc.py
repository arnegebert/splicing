from collections import defaultdict
import time
import numpy as np
from utils import reverse_complement, one_hot_encode_seq, intron_mean, exon_mean, intron_std, exon_std

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
save_to_low = 'hipsci_majiq/low.npy'
save_to_high = 'hipsci_majiq/high.npy'
save_to_cons = 'hipsci_majiq/cons.npy'

introns_bef_start = 70 # introns
exons_after_start = 70 # exons
exons_bef_end = 70 # exons
introns_after_end = 70 # introns

es_lines = []
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
with open('filtered_all.tsv') as f:
    loaded_chrom = 1
    chrom_seq = load_chrom_seq(loaded_chrom)

    for i, line in enumerate(f):
        if i == 0: continue
        genename, geneid, lsvid, epsi, stdev, lsvtype, njunc, nexon, denovojunc, \
        chrom, strand, juncoord, excoords, ircoord, ucsc = line.replace('\n', '').split('\t')
        denovo += denovojunc.count('1')


        coord1, coord2 = juncoord.split(';')
        idx1, idx2 = coord1.find('-'), coord2.find('-')
        s1, e1 = int(coord1[:idx1]), int(coord1[idx1 + 1:])
        s2, e2 = int(coord2[:idx2]), int(coord2[idx2 + 1:])

        excoord1, excoord2, excoord3 = excoords.split(';')
        exidx1, exidx2, exidx3 = excoord1.find('-'), excoord2.find('-'), excoord2.find('-')
        ex1start, ex1end = int(excoord1[:exidx1]), int(excoord1[exidx1 + 1:])
        ex2start, ex2end = int(excoord2[:exidx2]), int(excoord2[exidx2 + 1:])
        ex3start, ex3end = int(excoord3[:exidx3]), int(excoord3[exidx3 + 1:])

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
        if chrom > loaded_chrom:
            loaded_chrom += 1
            chrom_seq = load_chrom_seq(loaded_chrom)

        # standard: window1 will always be the junction skipping the cassette exon
        if (e1 - s1) > (e2 - s2):
            window_around_start1 = chrom_seq[s1 - introns_bef_start - 1:s1 + exons_after_start - 1]
            window_around_end1 = chrom_seq[e1 - exons_bef_end - 2:e1 + introns_after_end - 2]
            window_around_start2 = chrom_seq[s2 - introns_bef_start - 1:s2 + exons_after_start - 1]
            window_around_end2 = chrom_seq[e2 - exons_bef_end - 2:e2 + introns_after_end - 2]
            psi1, psi2 = map(float, epsi.split(';'))
        else:
            window_around_start1 = chrom_seq[s1 - introns_bef_start - 1:s1 + exons_after_start - 1]
            window_around_end1 = chrom_seq[e1 - exons_bef_end - 2:e1 + introns_after_end - 2]
            window_around_start2 = chrom_seq[s2 - introns_bef_start - 1:s2 + exons_after_start - 1]
            window_around_end2 = chrom_seq[e2 - exons_bef_end - 2:e2 + introns_after_end - 2]
            psi2, psi1 = map(float, epsi.split(';'))
        if strand == '-':
            window_around_start1, window_around_end1 = reverse_complement(window_around_end1[::-1]), \
                                                     reverse_complement(window_around_start1[::-1])
            window_around_start2, window_around_end2 = reverse_complement(window_around_end2[::-1]), \
                                                     reverse_complement(window_around_start2[::-1])
        # # encode & convert to numpy arrays
        start1, end1 = one_hot_encode_seq(window_around_start1), one_hot_encode_seq(window_around_end1)
        start1, end1 = np.array(start1), np.array(end1)
        start2, end2 = one_hot_encode_seq(window_around_start2), one_hot_encode_seq(window_around_end2)
        start2, end2 = np.array(start2), np.array(end2)

        # my brain is starting to hurt
        # don't think I need the below with standardization above
        # if (e1 - s1) > (e2 - s2): # s1-e1 is skipping junction
        #     l11, l21, l31 = ex1end-ex1start, ex3start-ex1end, ex3end-ex3start
        #     if s1 == s2: # left junction
        #         l12, l22, l32 = ex1end-ex1start, ex2start-ex1end, ex2end-ex2start
        #     elif e1 == e2: # right junction
        #         l12, l22, l32 = ex2end - ex2start, ex3start - ex2end, ex3end - ex3start
        # else: # s1-e1 is including junction
        #     if s1 == s2: # left junction
        #         l12, l22, l32 = ex1end-ex1start, ex2start-ex1end, ex2end-ex2start
        #     elif e1 == e2: # right junction
        #         l12, l22, l32 = ex2end - ex2start, ex3start - ex2end, ex3end - ex3start
        #
        #     l11, l21, l31 = ex1end-ex1start, ex3start-ex1end, ex3end-ex3start

        l11, l21, l31 = ex1end - ex1start, ex3start - ex1end, ex3end - ex3start
        if s1 == s2:  # left junction
            l12, l22, l32 = ex1end - ex1start, ex2start - ex1end, ex2end - ex2start
        elif e1 == e2:  # right junction
            l12, l22, l32 = ex2end - ex2start, ex3start - ex2end, ex3end - ex3start

        l11, l21, l31 = (l11 - exon_mean) / exon_std, (l21 - intron_mean) / intron_std, (l31 - exon_mean) / exon_std
        l12, l22, l32 = (l12 - exon_mean) / exon_std, (l22 - intron_mean) / intron_std, (l32 - exon_mean) / exon_std


        lens_and_psi_vector1 = np.array([l11, l21, l31, psi1])
        start_and_end1 = np.concatenate((start1, end1))
        sample1 = np.concatenate((start_and_end1,lens_and_psi_vector1.reshape(1,4))).astype(np.float32)
        if psi1 < 0.8:
            low_exons.append(sample1)
        elif psi1 < 1:
            high_exons.append(sample1)
        else:
            cons_exons.append(sample1)

        lens_and_psi_vector2 = np.array([l12, l22, l32, psi2])
        start_and_end2 = np.concatenate((start2, end2))
        sample2 = np.concatenate((start_and_end2,lens_and_psi_vector2.reshape(1,4))).astype(np.float32)
        if psi2 < 0.8:
            low_exons.append(sample2)
        elif psi2 < 1:
            high_exons.append(sample2)
        else:
            cons_exons.append(sample2)
        psis.append(psi1)
    low_psi_exons = np.array(low_exons)
    high_psi_exons = np.array(high_exons)
    cons_exons = np.array(cons_exons)

        # # save
        # es_lines.append((geneid, lsvid, lsvtype, epsi, stdev, a5ss, a3ss, es, njunc, nexon, juncoord, ircoord,
        #                  s1, e1, s2, e2))
        # if s1 == prev_start1:
        #     exons.append(es_lines[-1])
        # prev_start1 = s1


# es_lines.sort(key=lambda tup: tup[-1])
# es_lines.sort(key=lambda tup: tup[-2])
# es_lines.sort(key=lambda tup: tup[-3])
# es_lines.sort(key=lambda tup: tup[-4])
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

print(f'Number of junctions where cassette exon is left: {var1}') # 10886
print(f'Number of junctions where cassette exon is right: {var2}') # 10952
print(f'Number of junctions where cassette exon is neither left nor right: {var3}') # 862
print(f'Number of junctions where shorter junction is first: {shorter_first}') # 10899
print(f'Number of junctions where shorter junction is second: {shorter_second}') # 11801

print(f'Number of var1 junctions where shorter junction is first: {shorter_first_var1}') # 5379
print(f'Number of var1 junctions where shorter junction is second: {shorter_second_var1}') # 5507
print(f'Number of var2 junctions where shorter junction is first: {shorter_first_var2}') # 5076
print(f'Number of var2 junctions where shorter junction is second: {shorter_second_var2}') # 5876

print(f'Number of exon skipped events: {len(es_lines)}')  # 22700
for (k, v) in d.items():
    print(f'{k}: {v}')

# with open('sorted_all.psi.tsv', 'w') as f:
#     f.write(
#         f'geneid\tlsvid\tlsvtype\tE[psi]\tStd[E[psi]]\ta5ss\ta3ss\tES\tnjunc\tnexon\tjunc coord\tir coord\tstart1\tend1\tstart2\tend2\n')
#     for (geneid, lsvid, lsvtype, epsi, stdev, a5ss, a3ss, es, njunc, nexon, juncoord, ircoord,
#          s1, e1, s2, e2) in es_lines:
#         f.write(
#             f'{geneid}\t{lsvid}\t{lsvtype}\t{epsi}\t{stdev}\t{a5ss}\t{a3ss}\t{es}\t{njunc}\t{nexon}\t{juncoord}\t{ircoord}\t{s1}\t{e1}\t{s2}\t{e2}\n')

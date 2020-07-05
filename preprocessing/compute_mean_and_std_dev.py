import numpy as np

# exon_lens, intron_lens = [], []
# last_exon_start, last_exon_end = 0, 0
# with open('../data/gencode.v34.annotation.gtf') as f:
#     for i, line in enumerate(f):
#         if line[0] == '#': continue
#         if i % 1000 == 0: print(f'Line {i}')
#         chrom, src, tpe, start, end, _, _, _, _ = line.split('\t')
#         start, end = int(start), int(end)
#         if tpe == 'gene': continue
#         if tpe == 'transcript':
#             last_trans_start, last_trans_end = start, end
#             continue
#         elif tpe == 'exon':
#             if start == last_trans_start or end == last_trans_end: continue
#             exon_lens.append(end-start)
#
#             intron_lens.append(start-last_exon_end)
#             last_exon_start, last_exon_end = start, end
#
#
# exon_lens = np.array(exon_lens)
# print(f'Mean of internal exons: {np.mean(exon_lens)}')
# print(f'Std of internal exons: {np.std(exon_lens)}')
#
# intron_lens = np.array(intron_lens)
# print(f'Mean of introns: {np.mean(intron_lens)}')
# print(f'Std of introns: {np.std(intron_lens)}')

# Mean of internal exons: 145.4249515569803
# Std of internal exons: 197.9997431696026
# Mean of introns: -85.07464602577369
# Std of introns: 697114.5480490335

means = np.array([4736, 5883, 6375, 7168, 7277, 5961, 6703, 7354, 5351, 6412, 4341, 4570, 7351, 5653, 4660, 3661, 3193, 7905, 2032, 4403, 5086, 3924])
weights = np.array([19831, 11152, 12123, 7373, 8760, 10100, 20537, 6915, 7908, 9256, 10892, 11100, 3358, 6106, 7263, 8893, 11720, 2966, 10560, 5717, 2230, 4502])

weighted_mean = np.mean(means * weights)
normalized_mean = weighted_mean/np.mean(weights)
print(weighted_mean) #48359565.09090909
print(normalized_mean) # 5339.254007286889
print(np.mean(means)) # ~ 5454.5


std_dev = 17000 # estimated from https://www.researchgate.net/figure/tbl1_8491627
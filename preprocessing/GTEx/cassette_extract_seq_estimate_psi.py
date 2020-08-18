# given junction reads from the correct samples, extract the corresponding intron-exon sequences from the genome

# A LOT of this code is duplicated from extract_seq_and_estimate_psi_junc.py
import argparse
import csv
from timeit import default_timer as timer
from utils import exon_mean, exon_std, intron_mean, intron_std, assumed_read_length
import numpy as np

from utils import reverse_complement, one_hot_encode_seq

parser = argparse.ArgumentParser()
parser.add_argument('--tissue', type=str, default='', metavar='tissue',
                    help='type of tissue filtered for')
args = parser.parse_args()
tissue = 'heart' if not args.tissue else args.tissue

assert tissue in ['brain', 'cerebellum', 'heart']

startt = timer()
data_path = '../../data'
if tissue == 'brain':
    path_filtered_reads = f'{data_path}/gtex_processed/brain_cortex_junction_reads_one_sample.csv'
    path_highly_expr_genes = '../../data/gtex_processed/brain_cortex_tpm_one_sample.csv'
elif tissue == 'cerebellum':
    path_filtered_reads = f'{data_path}/gtex_processed/cerebellum_junction_reads_one_sample.csv'
    path_highly_expr_genes = '../../data/gtex_processed/cerebellum_tpm_one_sample.csv'
elif tissue == 'heart':
    path_filtered_reads = f'{data_path}/gtex_processed/heart_junction_reads_one_sample.csv'
    path_highly_expr_genes = '../../data/gtex_processed/heart_tpm_one_sample.csv'

save_to_low = f'gtex_processed/exon/{tissue}/low.npy'
save_to_high = f'gtex_processed/exon/{tissue}/high.npy'
save_to_cons = f'gtex_processed/exon/{tissue}/cons.npy'

print('-'*40)
print(f'Processing tissue type: {tissue}')
print('-'*40)

introns_bef_start = 70 # introns
exons_after_start = 70 # exons

exons_bef_end = 70 # exons
introns_after_end = 70 # introns

highly_expressed_genes = set()
def load_highly_expressed_genes():
    with open(path_highly_expr_genes) as f:
        for l in f:
            gene_id, tpm = l.split(',')
            highly_expressed_genes.add(gene_id)

load_highly_expressed_genes()

def contains_highly_expressed_gene(genes):
    for gene in genes:
        if gene in highly_expressed_genes:
            return True
    return False

gencode_genes = {}
# def load_gencode_genes():
#     with open(f'../../data/gencode_genes.csv') as f:
#         for line in f:
#             line = line.replace('\n','').split('\t')
#             if len(line) == 1: continue
#             gene, chr, start, end, strand = line[0], int(line[1][3:]), int(line[2]), int(line[3]), line[4]
#             if chr not in gencode_genes:
#                 gencode_genes[chr] = []
#             gencode_genes[chr].append((gene, start, end, strand))
#         print('Finished reading gencode genes')

def load_gencode_genes():
    with open(f'../../data/gencode_genes.csv') as f:
        for line in f:
            line = line.replace('\n','').split('\t')
            if len(line) == 1: continue
            gene, chr, start, end, strand = line[0], int(line[1][3:]), int(line[2]), int(line[3]), line[4]
            gencode_genes[gene] = strand
        print('Finished reading gencode genes')

load_gencode_genes()

def get_strand_based_on_gene(gene):
    if gene in gencode_genes:
        return gencode_genes[gene]
    else: return None

not_highly_expressed = 0
homeless_junctions = 0
current_idx_overlap = 0
too_short_exon = 0
gene_not_in_gencode = 0

def overlap(start, end, start2, end2):
    return not (end2 < start or start2 > end)

# want to load chromosome as one giant string
def load_chrom_seq(chrom):
    with open(f'{data_path}/chromosomes/chr{chrom}.fa') as f:
        loaded_chrom_seq = f.read().replace('\n', '')
        return loaded_chrom_seq.replace(f'<chr{chrom}','', 1)

def overlap(start, end, start2, end2):
    return not (end2 < start or start2 > end)

# want to load chromosome as one giant string
def load_chrom_seq(chrom):
    with open(f'{data_path}/chromosomes/chr{chrom}.fa') as f:
        loaded_chrom_seq = f.read().replace('\n', '')
        return loaded_chrom_seq.replace(f'<chr{chrom}','', 1)

junction_seqs = {}
junction_psis = {}
seqs_psis = {}
l1scons, l2scons, l3scons = [], [], []
l1scass, l2scass, l3scass = [], [], []
psis = []
cons_exons, high_exons, low_exons = [], [], []

with open(path_filtered_reads) as f:
    reader = csv.reader(f, delimiter="\n")
    # contains list with rows of samples
    junction_reads_file = list(reader)

with open(path_filtered_reads) as f:
    loaded_chrom = 1
    chrom_seq = load_chrom_seq(loaded_chrom)
    print('Start of iteration through lines')
    for i, line in enumerate(f):
        if i % 1000 == 0: # ~ 357500 junctions
            print(f'Reading line {i}')
        line = line.replace('\n', '').split(',')
        junction = line[0].split('_')
        try:
            read_chrom = int(junction[0].replace('chr',''))
        except ValueError:
            break
        start1, end1 = int(junction[1]), int(junction[2])

        # if chromosome changes, update loaded sequence until chromosome 22 reached
        if read_chrom > loaded_chrom:
            loaded_chrom += 1
            current_idx_overlap = 0
            chrom_seq = load_chrom_seq(loaded_chrom)

        """ Filtering """
        # a minimal length of 25nt for the exons and a length of 80nt for the neighboring introns are
        # required as exons/introns shorter than 25nt/80nt are usually caused by sequencing errors and
        # they represent less than 1% of the exon and intron length distributions
        # https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6722613/
        gene = line[2]
        in_highly_expressed_gene = contains_highly_expressed_gene([gene])
        if not in_highly_expressed_gene:
            not_highly_expressed += 1
            continue

        if end1 - start1 < 25:
            too_short_exon += 1
            continue

        # filter potential first and last exons in chromosome
        if i == 0 or i == len(junction_reads_file)-1: continue

        # heuristic: check the 10 junctions following the current junctions whether they overlap with current junction
        for idx_below in range(i + 1, i + 10):
            if idx_below >= len(junction_reads_file): break
            line2 = junction_reads_file[idx_below][0]
            junction2, count2, gene2 = line2.replace('\n','').split(',')
            _, start2, end2 = junction2.split('_')
            start2, end2 = int(start2), int(end2)
            if start1 == start2: # if junction starts at same point, check 10 following junctions
                for idx_below_below in range(idx_below + 1, idx_below + 10):
                    if idx_below_below >= len(junction_reads_file): break
                    line3 = junction_reads_file[idx_below_below][0]
                    junction3, count3, gene3 = line3.replace('\n','').split(',')
                    _, start3, end3 = junction3.split('_')
                    start3, end3 = int(start3), int(end3)
                    if end2 == end3 and start3 > end1: # junc 2 end at same place and junc 3 is to the right of junc 1
                        # cassette exon found
                        # start gives start of canonical nts -> -1
                        # end gives end of canonical nts -> -2

                        window_around_start = chrom_seq[end1 - introns_bef_start - 1:end1 + exons_after_start - 1]
                        window_around_end = chrom_seq[start3 - exons_bef_end - 2:start3 + introns_after_end - 2]
                        junction_seqs[line[0]] = [window_around_start, window_around_end]

                        strand = get_strand_based_on_gene(gene)
                        if not strand:
                            gene_not_in_gencode += 1
                            continue
                        if strand == '-':
                            window_around_start, window_around_end = reverse_complement(window_around_end[::-1]), \
                                                                     reverse_complement(window_around_start[::-1])
                        # almost always AG, but also frequently ac -2:0
                        # print(chrom_seq[b-2:b])
                        # almost always GT, but also many gc
                        # print(chrom_seq[c-1:c+1])

                        # read count from a -> b / c -> d
                        pos = int(count3) + int(line[1])
                        # read count from a -> d
                        neg = int(count2)

                        startw, endw = one_hot_encode_seq(window_around_start), one_hot_encode_seq(window_around_end)
                        startw, endw = np.array(startw), np.array(endw)
                        l1, l2, l3 = end1-start1, start3 - end1, end3 - start3
                        if (pos + neg) == 0: psi = 0
                        else:
                            norm_pos, norm_neg = pos / (l2 + assumed_read_length), neg / l2
                            psi = norm_pos / (norm_pos + norm_neg)
                        l1, l2, l3 = (l1-intron_mean)/intron_std, (l2-exon_mean)/exon_std, (l3-intron_mean)/intron_std

                        lens_and_psi_vector = np.array([l1, l2, l3, psi])
                        start_and_end = np.concatenate((startw, endw))
                        sample = np.concatenate((start_and_end, lens_and_psi_vector.reshape(1, 4)))
                        sample = sample.astype(np.float32)
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
psis = np.array(psis)

print(f'Number of samples: {len(psis)}')
print(f'Mean PSI: {np.mean(psis)}')
print(f'Median PSI: {np.median(psis)}')

print(f'Number of generated training samples: {len(low_psi_exons)+len(high_psi_exons)+len(cons_exons)}')  # 22700

print(f'Number of low PSI exons: {len(low_exons)}')
print(f'Number of high PSI exons: {len(high_exons)}')
print(f'Number of cons exons: {len(cons_exons)}')

print(f'Number of too short exons: {too_short_exon}')
print(f'Number of skipped homeless junctions: {homeless_junctions} ')
print(f'Number of junctions skipped because not part of highly expressed gene {not_highly_expressed}')
print(f'Number of junctions after filtering: {len(seqs_psis)}')
print(f'Number of samples from genes not found in gencode: {gene_not_in_gencode}')


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

l1s, l2s, l3s = np.concatenate((l1scons, l1scass)), np.concatenate((l2scons, l2scass)), np.concatenate((l3scons, l3scass))
l1avg, l2avg, l3avg = np.average(l1s), np.average(l2s), np.average(l3s)
l1med, l2med, l3med = np.median(l1s), np.median(l2s), np.median(l3s)
l1std, l2std, l3std = np.std(l1s), np.std(l2s), np.std(l3s)

print(f'Cons + Cass')
print(f'L1avg: {l1avg} +- {l1std}, l2avg: {l2avg} +- {l2std}, l3avg: {l3avg} +- {l3std}')
print(f'L1 median: {l1med}, l2 median: {l2med}, l3 median: {l3med}')

np.save(f'{data_path}/{save_to_low}', low_exons)
np.save(f'{data_path}/{save_to_high}', high_exons)
np.save(f'{data_path}/{save_to_cons}', cons_exons)

print('Processing finished')
endt = timer()

print(f'It took {endt-startt} s to generate the data sets')
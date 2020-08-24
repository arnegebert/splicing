# given junction reads from the correct samples, extract the corresponding intron-exon sequences from the genome
import csv
import linecache
from timeit import default_timer as timer
import numpy as np
import argparse

from utils import reverse_complement, one_hot_encode_seq, intron_mean, intron_std

parser = argparse.ArgumentParser()
parser.add_argument('--tissue', type=str, default='', metavar='tissue',
                    help='type of tissue filtered for')
args = parser.parse_args()

startt = timer()
tissue = 'cerebellum' if not args.tissue else args.tissue

assert tissue in ['brain', 'cerebellum', 'heart']
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

save_to_low = f'gtex_processed/junc/{tissue}/low.npy'
save_to_high = f'gtex_processed/junc/{tissue}/high.npy'
save_to_cons = f'gtex_processed/junc/{tissue}/cons.npy'

print('-'*40)
print(f'Processing tissue type: {tissue}')
print('-'*40)

introns_bef_start = 70 # introns
exons_after_start = 70 # exons

exons_bef_end = 70 # exons
introns_after_end = 70 # introns

highly_expressed_genes = dict()
def load_highly_expressed_genes():
    with open(path_highly_expr_genes) as f:
        for l in f:
            gene_id, tpm = l.split(',')
            highly_expressed_genes[gene_id] = tpm

load_highly_expressed_genes()

def contains_highly_expressed_gene(genes):
    for gene in genes:
        if gene in highly_expressed_genes:
            return True, highly_expressed_genes[gene]
    return False, 0

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
too_short_intron = 0
gene_not_in_gencode = 0

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
l1_lens = []
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
        line = line.replace('\n','').split(',')
        junction = line[0].split('_')
        try:
            read_chrom = int(junction[0][3:])
        except ValueError:
            break
        start, end = int(junction[1]), int(junction[2])
        read_count = int(line[1])
        # if chromosome changes, update loaded sequence until chromosome 22 reached
        if read_chrom > loaded_chrom:
            loaded_chrom += 1
            current_idx_overlap = 0
            chrom_seq = load_chrom_seq(loaded_chrom)

        # extract sequence around start
            # however, not extremly big priorities as it essentially just some data pollution
        # q: does it work to just remove the first and last exon boundary found for each chromosome?
            # a: no, because that doesn't solve the problems for genes

        """ Filtering """
        # a minimal length of 25nt for the exons and a length of 80nt for the neighboring introns are
        # required as exons/introns shorter than 25nt/80nt are usually caused by sequencing errors and
        # they represent less than 1% of the exon and intron length distributions
        # https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6722613/
        # genes, updated_idx = map_from_position_to_genes(loaded_chrom, start, end, current_idx_overlap)
        # if not genes: homeless_junctions += 1
        # current_idx_overlap = updated_idx
        gene = line[2]
        in_highly_expressed_gene = contains_highly_expressed_gene([gene])
        if not in_highly_expressed_gene:
            not_highly_expressed += 1
            continue

        if end - start < 80:
            too_short_intron += 1
            continue

        # make sure there are at least 'introns_bef_start' intron nts between exon junctions
        # q: do i even want to do this? how do i do this?


        # make sure that very first exon in chromosome doesn't contain N nt input
        # make sure that very last exon in chromosome doesn't contain N nt input
        if i == 0 or i == len(junction_reads_file)-1:
            print('----------------------------------------------------------------------')
            continue

        # if start - introns_bef_start < prev_end:
        #     continue

        # remove first exon in gene

        # gene_start = 0
        # gene_end = 1e10
        # remove last exon in gene
        # if end + introns_after_end > gene_end:
        #     continue
        # todo: probably want to make sure that i dont have junctions where distance between
        # them is smaller than flanking sequence i extract

        """ Extraction of the sequence """
        # start gives start of canonical nts -> -1
        # end gives end of canonical nts -> -2
        window_around_start = chrom_seq[start-introns_bef_start-1:start+exons_after_start-1]
        window_around_end = chrom_seq[end-exons_bef_end-2:end+introns_after_end-2]
        junction_seqs[0] = [window_around_start, window_around_end]

        strand = get_strand_based_on_gene(gene)
        if not strand:
            gene_not_in_gencode += 1
            continue
        if strand == '-':
            window_around_start, window_around_end = reverse_complement(window_around_end[::-1]), \
                                                     reverse_complement(window_around_start[::-1])

        """ Estimation of the PSI value """
        # PSI = pos / (pos + neg)
        pos = read_count
        # neg == it overlaps with the junction in any way:

        # check all junctions above until end2 < start
        # good idea, but doesn't work because you can have
        # 1 -> 10
        # 2 -> 3
        # 4 -> 5
        # -> changing to look at 10 rows above (as heuristic)
        neg = 0
        for idx_above in range(i-1, i-10, -1):
            if idx_above <= 0: break
            # [0] because it's in a one element list iirc
            line2 = junction_reads_file[idx_above][0]
            line2 = line2.split(',')
            junction2 = line2[0].split('_')
            start2, end2 = int(junction2[1]), int(junction2[2])
            if end2 >= start:
                neg += int(line2[1])

        # check all junctions below until start2 > end
        for idx_below in range(i+1, i+10):
            if idx_below >= len(junction_reads_file): break
            line2 = junction_reads_file[idx_below][0]
            line2 = line2.split(',')
            junction2 = line2[0].split('_')
            start2, end2 = int(junction2[1]), int(junction2[2])
            if end >= start2:
                neg += int(line2[1])

        if pos + neg == 0: psi = 0
        else: psi = pos / (pos + neg)

        startw, endw = one_hot_encode_seq(window_around_start), one_hot_encode_seq(window_around_end)
        startw, endw = np.array(startw), np.array(endw)
        l1, l2, l3 = 0, end - start, 0
        l2 = (l2 - intron_mean) / intron_std

        lens_and_psi_vector = np.array([l1, l2, l3, psi])
        start_and_end = np.concatenate((startw, endw))
        sample = np.concatenate((start_and_end,lens_and_psi_vector.reshape(1,4)))
        sample = sample.astype(np.float32)
        if psi < 0.8:
            low_exons.append(sample)
        elif psi < 1:
            high_exons.append(sample)
        else:
            cons_exons.append(sample)
        psis.append(psi)
        junction_psis[line[0]] = psi
        l1_lens.append(end-start)
        seqs_psis[line[0]] = (window_around_start, window_around_end, psi)

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

print(f'Number of too short introns: {too_short_intron}')
print(f'Number of skipped homeless junctions: {homeless_junctions} ')
print(f'Number of junctions skipped because not part of highly expressed gene {not_highly_expressed}')
print(f'Number of junctions after filtering: {len(seqs_psis)}')
print(f'Number of samples from genes not found in gencode: {gene_not_in_gencode}')

# chrom 1 to 10:
# 9023.466260727668
# 27015.74031545284

# chrom 1 to 22:
# 7853.118899261425
# 23917.691461462917

l1_lens = np.array(l1_lens)
avg_len = np.mean(l1_lens)
std_len = np.std(l1_lens)

print(f'Average length of l1: {avg_len}')
print(f'Standard deviation of l1: {std_len}')

np.save(f'{data_path}/{save_to_low}', low_exons)
np.save(f'{data_path}/{save_to_high}', high_exons)
np.save(f'{data_path}/{save_to_cons}', cons_exons)

# with open(f'{data_path}/{save_to}', 'w') as f:
#     f.write(f'{avg_len},{std_len}\n')
#     print('Beginning to write estimated PSIs and extracted sequences')
#     for junction, (start_seq, end_seq, psi) in seqs_psis.items():
#         f.write(f'{junction},{start_seq},{end_seq},{psi}\n')

print('Processing finished')
endt = timer()

print(f'It took {endt-startt} s to generate the data sets')
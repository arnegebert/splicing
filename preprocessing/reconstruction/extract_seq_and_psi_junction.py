# given junction reads from the correct samples, extract the corresponding intron-exon sequences from the genome
import csv
import linecache
from timeit import default_timer as timer
import numpy as np
import matplotlib.pyplot as plt

from utils import reverse_complement, one_hot_encode_seq, intron_std, intron_mean

startt = timer()
data_path = '../../data'
path_filtered_reads = f'{data_path}/gtex_processed/brain_cortex_junction_reads_one_sample.csv'

save_to_low = 'dsc_reconstruction_junction/low.npy'
save_to_high = 'dsc_reconstruction_junction/high.npy'
save_to_cons = 'dsc_reconstruction_junction/cons.npy'
last_chrom = 24

introns_bef_start = 70 # introns
exons_after_start = 70 # exons

exons_bef_end = 70 # exons
introns_after_end = 70 # introns

highly_expressed_genes = set()
def load_highly_expressed_genes():
    with open(f'{data_path}/gtex_processed/brain_cortex_tpm_one_sample.csv') as f:
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
def load_gencode_genes():
    with open(f'{data_path}/gencode_genes.csv') as f:
        for line in f:
            line = line.split('\t')
            if len(line) == 1: continue
            gene, chr, start, end = line[0], int(line[1][3:]), int(line[2]), int(line[3][:-1])
            if chr not in gencode_genes:
                gencode_genes[chr] = []
            gencode_genes[chr].append((gene, start, end))
        print('Finished reading gencode genes')

load_gencode_genes()

not_highly_expressed = 0
homeless_junctions = 0
current_idx_overlap = 0
too_short_intron = 0
non_dsc_junction = 0

def map_from_position_to_genes(chr, start, end, idx):
    gene_start_and_ends = gencode_genes[chr]
    while gene_start_and_ends[idx][1] <= end and idx < len(gene_start_and_ends)-1:
        idx += 1

    overlapping_genes = []
    for i in range(idx-1, idx-11, -1):
        if i < 0: break
        gene, start2, end2 = gene_start_and_ends[i]
        if overlap(start, end, start2, end2):
            overlapping_genes.append(gene)
    # if len(overlapping_genes) == 0:
    #     print(f'this mofo belongs to no gene')
    return overlapping_genes, idx

def overlap(start, end, start2, end2):
    return not (end2 < start or start2 > end)

# want to load chromosome as one giant string
def load_chrom_seq(chrom):
    with open(f'{data_path}/chromosomes/chr{chrom}.fa') as f:
        loaded_chrom_seq = f.read().replace('\n', '')
        return loaded_chrom_seq.replace(f'<chr{chrom}','', 1)

def load_DSC_exons():
    with open(f'../../data/dsc_reconstruction_junction/cons_exons.csv') as f:
        reader_cons = csv.reader(f, delimiter='\t')
        cons = list(reader_cons)
        cons_divided = {}
        for chrom, strand, start, end, count, skip, constit_level, start_seq, end_seq, l1, l2, l3 in cons:
            chrom, start, end, count, skip, constit_level = int(chrom[3:]), int(start), int(end), int(count),\
                                                            int(skip), float(constit_level)
            if chrom not in cons_divided:
                cons_divided[chrom] = []
            cons_divided[chrom].append((strand, start, end, count, skip, constit_level))
    with open(f'../../data/dsc_reconstruction_junction/cass_exons.csv') as f:
        reader_cass = csv.reader(f, delimiter='\t')
        cass = list(reader_cass)
        cass_divided = {}
        for chrom, strand, start, end, count, skip, constit_level, start_seq, end_seq, l1, l2, l3 in cass:
            chrom, start, end, count, skip, constit_level = int(chrom[3:]), int(start), int(end), int(count),\
                                                            int(skip), float(constit_level)
            if chrom not in cass_divided:
                cass_divided[chrom] = []
            cass_divided[chrom].append((strand, start, end, count, skip, constit_level))
    return cons_divided, cass_divided

DSC_cons, DSC_cass = load_DSC_exons()
print('Loaded DSC exons')

# -1, otherwise no junction start / exon end matches
def find_DSC_exon_belonging_to_junction(chrom, junc_start, junc_end):
    cons, cass = DSC_cons[chrom], DSC_cass[chrom]
    for strand, exon_start, exon_end, count, skip, constit_level in cons:
        if (junc_start-1 == exon_end or junc_end == exon_start):
            return (chrom, strand, exon_start, exon_end, count, skip, constit_level)

    for strand, exon_start, exon_end, count, skip, constit_level in cass:
        if (junc_start-1 == exon_end or junc_end == exon_start):
            return (chrom, strand, exon_start, exon_end, count, skip, constit_level)
    raise ValueError('No matching exon for junction in DSC data')

# no junction start / exon end ever
def find_DSC_exon_belonging_to_junction2(chrom, junc_start, junc_end):
    cons, cass = DSC_cons[chrom], DSC_cass[chrom]
    for strand, exon_start, exon_end, count, skip, constit_level in cons:
        if (junc_start-1 == exon_end):
            print('cons junction start / exon end match')
            return (chrom, strand, exon_start, exon_end, count, skip, constit_level)

    for strand, exon_start, exon_end, count, skip, constit_level in cons:
        if (junc_end == exon_start):
            print('cons junction end / exon start match')
            return (chrom, strand, exon_start, exon_end, count, skip, constit_level)

    for strand, exon_start, exon_end, count, skip, constit_level in cass:
        if (junc_end == exon_start):
            print('cass junction end / exon start match')

            return (chrom, strand, exon_start, exon_end, count, skip, constit_level)
    for strand, exon_start, exon_end, count, skip, constit_level in cass:
        if (junc_start-1 == exon_end):
            print('cass junction start / exon end match')
            return (chrom, strand, exon_start, exon_end, count, skip, constit_level)
    raise ValueError('No matching exon for junction in DSC data')


psis_gtex, psis_dsc = [], []

with open(path_filtered_reads) as f:
    reader = csv.reader(f, delimiter="\n")
    # contains list with rows of samples
    junction_reads_file = list(reader)

l2s = []
cons_exons, high_psi_exons, low_psi_exons = [], [], []
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
        if read_chrom == last_chrom: break
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

        try:
            chrom, strand, exon_start, exon_end, count, skip, constit_level = \
                find_DSC_exon_belonging_to_junction(loaded_chrom, start, end)

        except ValueError:
            non_dsc_junction += 1
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

        # todo stand switching
        if strand == '-':
            window_around_start, window_around_end = reverse_complement(window_around_end[::-1]), \
                                                     reverse_complement(window_around_start[::-1])
        # almost always GT, but also many gc
        # print(chrom_seq[start-1:start+1])
        # almost always AG or AC
        # print(chrom_seq[end - 2:end + 0])

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
        l1, l2, l3 = 0, end-start, 0
        l2 = (l2-intron_mean)/intron_std
        lens_and_psi_vector = np.array([l1, l2, l3, psi])
        start_and_end = np.concatenate((startw, endw))
        sample = np.concatenate((start_and_end,lens_and_psi_vector.reshape(1,4))).astype(np.float32)
        if psi < 0.8:
            low_psi_exons.append(sample)
            l2s.append(l2)
        elif psi < 1:
            high_psi_exons.append(sample)
            l2s.append(l2)
        else:
            cons_exons.append(sample)
            l2s.append(l2)

        psis_dsc.append(constit_level)
        psis_gtex.append(psi)


print(f'Number of too short introns: {too_short_intron}')
print(f'Number of skipped homeless junctions: {homeless_junctions} ')
print(f'Number of junctions skipped because not part of highly expressed gene {not_highly_expressed}')
print(f'Number of skipped non-dsc junctions: {non_dsc_junction}')

print(f'Number of low PSI junctions: {len(low_psi_exons)}')
print(f'Number of high PSI junctions: {len(high_psi_exons)}')
print(f'Number of cons junctions: {len(cons_exons)}')
print(f'Total number of junctions: {len(low_psi_exons) + len(high_psi_exons) + len(cons_exons)}')

np.save(f'{data_path}/{save_to_low}', low_psi_exons)
np.save(f'{data_path}/{save_to_high}', high_psi_exons)
np.save(f'{data_path}/{save_to_cons}', cons_exons)

# chrom 1 to 10:
# 9023.466260727668
# 27015.74031545284

# chrom 1 to 22:
# 7853.118899261425
# 23917.691461462917

l2s = np.array(l2s)
avg_len = np.mean(l2s)
std_len = np.std(l2s)



psis_dsc, psis_gtex = np.array(psis_dsc), np.array(psis_gtex)
plt.hist(psis_gtex)
plt.title('PSI Value distribution DSC-like Junction dataset')
plt.xlabel('PSI value')
plt.ylabel('number of data points')
plt.show()

print('----------------------------------')

print(f'Average PSI value from DSC dataset: {np.mean(psis_dsc)}')
print(f'Average PSI value from GTEx dataset: {np.mean(psis_gtex)}')
print(f'Median PSI value from DSC dataset: {np.median(psis_dsc)}')
print(f'Median PSI value from GTEx dataset: {np.median(psis_gtex)}')
print(f'Correlation between DSC and GTEx PSI values: {np.corrcoef(psis_dsc, psis_gtex)[0,1]}')
print(f'Average absolute difference between DSC and GTEx PSI values: {np.mean(np.abs(psis_dsc-psis_gtex))}')
print('---------------------------------')
print(f'Average length of l2: {avg_len}')
print(f'Standard deviation of l2: {std_len}')

print('Processing finished')
endt = timer()

print(f'It took {endt-startt} s to generate the data sets')
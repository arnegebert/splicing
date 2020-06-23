# given junction reads from the correct samples, extract the corresponding intron-exon sequences from the genome
import csv
import linecache
from timeit import default_timer as timer
import numpy as np
from utils import reverse_complement, one_hot_encode_seq

startt = timer()
data_path = '../../data'
path_filtered_reads = f'{data_path}/gtex_processed/brain_cortex_junction_reads_one_sample.csv'
save_to_low = 'dsc_reconstruction_exon/brain_cortex_full_low.npy'
save_to_high = 'dsc_reconstruction_exon/brain_cortex_full_high.npy'
save_to_cons = 'dsc_reconstruction_exon/brain_cortex_full_cass.npy'
last_chrom = 23

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


# want to load chromosome as one giant string
def load_chrom_seq(chrom):
    with open(f'{data_path}/chromosomes/chr{chrom}.fa') as f:
        loaded_chrom_seq = f.read().replace('\n', '')
        # reader = csv.reader(f, delimiter=" ")
        # lines = list(reader)
        # complete = ''.join(lines)
        # contains list with rows of samples
        # log10 solution would be cleaner, but this is more readable
        # cutting out '>chr<chrom>'
        if chrom < 10:
            return loaded_chrom_seq[5:]
        else:
            return loaded_chrom_seq[6:]

def load_DSC_exons():
    with open(f'../../data/dsc_reconstruction_junction/cons_exons.csv') as f:
        reader_cons = csv.reader(f, delimiter='\t')
        cons = list(reader_cons)
        cons_divided = {}
        for chrom, strand, start, end, count, skip, constit_level, l1, l2, l3 in cons:
            chrom, start, end, count, skip, constit_level, l1, l2, l3 = int(chrom[3:]), int(start), int(end), \
                                                                        int(count), int(skip), float(constit_level), \
                                                                        float(l1), float(l2), float(l3)
            if chrom not in cons_divided:
                cons_divided[chrom] = []
            cons_divided[chrom].append((chrom, strand, start, end, count, skip, constit_level, l1, l2, l3))
    with open(f'../../data/dsc_reconstruction_junction/cass_exons.csv') as f:
        reader_cass = csv.reader(f, delimiter='\t')
        cass = list(reader_cass)
        cass_divided = {}
        for chrom, strand, start, end, count, skip, constit_level, l1, l2, l3 in cass:
            chrom, start, end, count, skip, constit_level, l1, l2, l3 = int(chrom[3:]), int(start), int(end), \
                                                                        int(count), int(skip), float(constit_level), \
                                                                        float(l1), float(l2), float(l3)
            if chrom not in cass_divided:
                cass_divided[chrom] = []
            cass_divided[chrom].append((chrom, strand, start, end, count, skip, constit_level, l1, l2, l3))
    return cons_divided, cass_divided

DSC_cons, DSC_cass = load_DSC_exons()
print('Loaded DSC exons')

def overlap(start, end, start2, end2):
    return not (start > end2 or end < start2)

def initialize_DSC_exon_counts(DSC_cons, DSC_cass):
    cons_counts, cass_counts = dict(), dict()
    for chrom, strand, start, end, count, skip, constit_level, l1, l2, l3 in DSC_cons.values():
        cons_counts[(start, end)] = [0, 0, (chrom, strand, start, end, count, skip, constit_level, l1, l2, l3)]
    for chrom, strand, start, end, count, skip, constit_level, l1, l2, l3 in DSC_cass.values():
        cass_counts[(start, end)] = [0, 0, (chrom, strand, start, end, count, skip, constit_level, l1, l2, l3)]
    return cons_counts, cass_counts

DSC_cons_counts, DSC_cass_counts = initialize_DSC_exon_counts(DSC_cons, DSC_cass)

# -1, otherwise no junction start / exon end matches
def add_junction_reads_to_DSC_exon_counts(chrom, junc_start, junc_end, reads):
    cons, cass = DSC_cons[chrom], DSC_cass[chrom]
    for exon_chr, strand, exon_start, exon_end, count, skip, constit_level in cons:
        if junc_start-1 == exon_end or junc_end == exon_start:
            DSC_cons_counts[(exon_start, exon_end)][0] += reads
        elif overlap(exon_start, exon_end, junc_start-1, junc_end):
            DSC_cons_counts[(exon_start, exon_end)][1] += reads

    for exon_chr, strand, exon_start, exon_end, count, skip, constit_level in cass:
        if junc_start-1 == exon_end or junc_end == exon_start:
            if junc_start - 1 == exon_end or junc_end == exon_start:
                DSC_cass_counts[(exon_start, exon_end)][0] += reads
            elif overlap(exon_start, exon_end, junc_start-1, junc_end):
                DSC_cass_counts[(exon_start, exon_end)][1] += reads

seqs_psis = {}
l1_lens = []


with open(path_filtered_reads) as f:
    reader = csv.reader(f, delimiter="\n")
    # contains list with rows of samples
    junction_reads_file = list(reader)

with open(path_filtered_reads) as f:
    print('Start of iteration through lines')
    prev_chrom = 1
    for i, line in enumerate(f):
        if i % 1000 == 0: # ~ 357500 junctions
            print(f'Reading line {i}')
        line = line.split(',')
        junction = line[0].split('_')
        try:
            read_chrom = int(junction[0][3:])
        except ValueError:
            break
        if read_chrom == last_chrom: break
        if read_chrom != prev_chrom:
            current_idx_overlap = 0
            prev_chrom = read_chrom
        start, end = int(junction[1]), int(junction[2])
        read_count = int(line[1][:-1])

        """ Filtering """
        # a minimal length of 25nt for the exons and a length of 80nt for the neighboring introns are
        # required as exons/introns shorter than 25nt/80nt are usually caused by sequencing errors and
        # they represent less than 1% of the exon and intron length distributions
        # https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6722613/
        genes, updated_idx = map_from_position_to_genes(read_chrom, start, end, current_idx_overlap)
        if not genes: homeless_junctions += 1
        current_idx_overlap = updated_idx
        in_highly_expressed_gene = contains_highly_expressed_gene(genes)
        if not in_highly_expressed_gene:
            not_highly_expressed += 1
            continue

        if end - start < 80:
            too_short_intron += 1
            continue

        if i == 0 or i == len(junction_reads_file)-1:
            print('----------------------------------------------------------------------')
            continue
        """Accumulating"""
        add_junction_reads_to_DSC_exon_counts(read_chrom, start, end, read_count)

no_junction = 0
psis_gtex, psis_dsc = [], []

"""Going through exons after all junction reads have been accounted for"""
def encoding_and_sequence_extraction(DSC_counts):
    cons_exons, high_psi_exons, low_psi_exons = [], [], []
    loaded_chrom = 1
    chrom_seq = load_chrom_seq(loaded_chrom)
    no_junction = 0
    for pos, neg, (read_chrom, strand, start, end, count, skip, constit_level, l1, l2, l3) in DSC_counts:
        if pos == 0 and neg == 0:
            no_junction += 1
            continue
        if read_chrom > loaded_chrom:
            loaded_chrom += 1
            chrom_seq = load_chrom_seq(loaded_chrom)

        psi = pos / (pos + neg)
        psis_gtex.append(psi); psis_dsc.append(constit_level)

        window_around_start = chrom_seq[start-introns_bef_start-1:start+exons_after_start-1]
        window_around_end = chrom_seq[end-exons_bef_end-2:end+introns_after_end-2]
        if strand == '-':
            window_around_start = reverse_complement(window_around_start)
            window_around_end = reverse_complement(window_around_end)
        start, end = one_hot_encode_seq(window_around_start), one_hot_encode_seq(window_around_end)
        dummy_vector = [0] * len(start)
        dummy_vector[:4] = l1, l2, l3, psi
        if psi < 0.8:
            low_psi_exons.append([start, end, dummy_vector])
        elif psi < 1:
            high_psi_exons.append([start, end, dummy_vector])
        else:
            cons_exons.append([start, end, dummy_vector])

    print(f'Skipped DSC exons because no junctions contributed any reads: {no_junction}')
    low_psi_exons = np.array(low_psi_exons)
    low_psi_exons[:, 2, :] = low_psi_exons[:, 2, :].astype(np.float32)
    high_psi_exons = np.array(high_psi_exons)
    high_psi_exons[:, 2, :] = high_psi_exons[:, 2, :].astype(np.float32)
    cons_exons = np.array(cons_exons)
    cons_exons[:, 2, :] = cons_exons[:, 2, :].astype(np.float32)

    return low_psi_exons, high_psi_exons, cons_exons

low_cons_exons, high_cons_exons, cons_cons_exons = encoding_and_sequence_extraction(DSC_cons_counts)
low_cass_exons, high_cass_exons, cons_cass_exons = encoding_and_sequence_extraction(DSC_cons_counts)

low_exons = np.concatenate((low_cons_exons, low_cass_exons))
high_exons = np.concatenate((high_cons_exons, high_cass_exons))
cons_exons = np.concatenate((cons_cons_exons, cons_cass_exons))

print(f'Number of low PSI exons: {len(low_exons)}')
print(f'Number of high PSI exons: {len(high_exons)}')
print(f'Number of cons exons: {len(cons_exons)}')
print(f'Total number of exons: {len(low_exons) + len(high_exons) + len(cons_exons)}')
# pytorch loss expects float with 32 bits, otherwise we will have label with 64 bits

np.save(f'{data_path}/{save_to_low}', low_exons)
np.save(f'{data_path}/{save_to_high}', high_exons)
np.save(f'{data_path}/{save_to_cons}', cons_exons)

# chrom 1 to 10:
# 9023.466260727668
# 27015.74031545284

# chrom 1 to 22:
# 7853.118899261425
# 23917.691461462917

psis_dsc, psis_gtex = np.array(psis_dsc), np.array(psis_gtex)
print('----------------------------------')
print(f'Average PSI value from DSC dataset: {np.mean(psis_dsc)}')
print(f'Average PSI value from GTEx dataset: {np.mean(psis_gtex)}')
print(f'Median PSI value from DSC dataset: {np.median(psis_dsc)}')
print(f'Median PSI value from GTEx dataset: {np.median(psis_gtex)}')
print(f'Correlation between DSC and GTEx PSI values: {np.corrcoef(psis_dsc, psis_gtex)[0,1]}')
print(f'Average absolute differenec between DSC and GTEx PSI values: {np.mean(np.abs(psis_dsc-psis_gtex))}')
print('---------------------------------')


print('Processing finished')
endt = timer()

print(f'It took {endt-startt} s to generate the data sets')
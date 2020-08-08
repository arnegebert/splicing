# given junction reads from the correct samples, extract the corresponding intron-exon sequences from the genome
import csv
import linecache
from timeit import default_timer as timer
import numpy as np

startt = timer()
data_path = '../../data'
path_filtered_reads = f'{data_path}/gtex_processed/brain_cortex_junction_reads_one_sample.csv'
save_to = 'gtex_processed/brain_cortex_cassette_full.csv'
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
too_short_exon = 0

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

junction_seqs = {}
junction_psis = {}
seqs_psis = {}
exon_lens = []

with open(path_filtered_reads) as f:
    reader = csv.reader(f, delimiter="\n")
    # contains list with rows of samples
    junction_reads_file = list(reader)

with open(path_filtered_reads) as f:
    loaded_chrom = 1
    chrom_seq = load_chrom_seq(loaded_chrom)
    # all = list(f)
    print('Start of iteration through lines')
    for i, line in enumerate(f):
        if i % 1000 == 0: # ~ 357500 junctions
            print(f'Reading line {i}')
        line = line.replace('\n', '').split(',')
        # split_idx = line.find(',')
        junction = line[0].split('_')
        try:
            read_chrom = int(junction[0][3:])
        except ValueError:
            break
        if read_chrom == last_chrom: break
        a, b = int(junction[1]), int(junction[2])
        start, end = int(junction[1]), int(junction[2])

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
        # genes, updated_idx = map_from_position_to_genes(loaded_chrom, start, end, current_idx_overlap)
        # if not genes: homeless_junctions += 1
        # current_idx_overlap = updated_idx
        gene = line[2]
        in_highly_expressed_gene = contains_highly_expressed_gene([gene])
        if not in_highly_expressed_gene:
            not_highly_expressed += 1
            continue

        if end - start < 25:
            too_short_exon += 1
            continue

        # make sure there are at least 'introns_bef_start' intron nts between exon junctions
        # q: do i even want to do this? how do i do this?


        # make sure that very first exon in chromosone doesn't contain N nt input
        # make sure that very last exon in chromosone doesn't contain N nt input
        if i == 0 or i == len(junction_reads_file)-1: continue


        for idx_below in range(i + 1, i + 10):
            if idx_below >= len(junction_reads_file): break
            line2 = junction_reads_file[idx_below][0]
            junction2, count2 = line2.replace('\n','').split(',')
            _, a2, d = junction2.split('_')
            a2, d = int(a2), int(d)
            if a == a2:
                for idx_below_below in range(idx_below + 1, idx_below + 10):
                    if idx_below_below >= len(junction_reads_file): break
                    line2 = junction_reads_file[idx_below_below][0]
                    junction2, count3 = line2.replace('\n','').split(',')
                    _, c, d2 = junction2.split('_')
                    c, d2 = int(c), int(d2)
                    if d == d2 and c > b:
                        # start gives start of canonical nts -> -1
                        # end gives end of canonical nts -> -2
                        # todo: probably switch boundary adjustments
                        window_around_start = chrom_seq[b - introns_bef_start - 1:b + exons_after_start - 1]
                        window_around_end = chrom_seq[c - exons_bef_end - 2:c + introns_after_end - 2]
                        junction_seqs[line[0]] = [window_around_start, window_around_end]

                        # almost always AG, but also frequently ac -2:0
                        # print(chrom_seq[b-2:b])
                        # almost always GT, but also many gc
                        # print(chrom_seq[c-1:c+1])

                        # read count from a -> b / c -> d
                        pos = int(count3) + int(line[1])
                        # read count from a -> d
                        neg = int(count2)
                        if (pos + neg) == 0: psi = 0
                        else: psi = pos / (pos + neg)

                        l2 = c-b
                        exon_lens.append(l2)
                        seqs_psis[line[0]] = (window_around_start, window_around_end, psi)

print(f'Number of skipped homeless junctions: {homeless_junctions} ')
print(f'Number of junctions skipped because not part of highly expressed gene {not_highly_expressed}')
print(f'Number of cassette exons after filtering: {len(seqs_psis)}')

exon_lens = np.array(exon_lens)
# intron_lens = np.array(intron_lens)
exon_mean, exon_std = np.mean(exon_lens), np.std(exon_lens)
# intron_mean, intron_std = np.mean(intron_lens), np.std(intron_lens)
print(f'Exon mean: {exon_mean}')
print(f'Exon std: {exon_std}')
# print(f'Intron mean: {intron_mean}')
# print(f'Intron std: {intron_std}')


with open(f'{data_path}/{save_to}', 'w') as f:
    print('Beginning to write estimated PSIs and extracted sequences')
    for junction, (start_seq, end_seq, psi) in seqs_psis.items():
        f.write(f'{junction},{start_seq},{end_seq},{psi}\n')

print('Processing finished')
endt = timer()

print(f'It took {endt-startt} s to generate the data sets')
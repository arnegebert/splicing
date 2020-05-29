# given junction reads from the correct samples, extract the corresponding intron-exon sequences from the genome
import csv

data_path = '../data'
path_filtered_reads = f'{data_path}/brain_cortex_junction_reads.csv'

# todo do naive PSI estimation


introns_bef_start = 50 # introns
exons_after_start = 30 # exons

exons_bef_end = 30 # exons
introns_after_end = 50 # introns

def load_chrom_seq(chrom):
    with open(f'{data_path}/chr{chrom}') as f:
        reader = csv.reader(f, delimiter="\t")
        # contains list with rows of samples
        return list(reader)

junction_seqs = {}

with open(path_filtered_reads) as f:
    loaded_chrom = 1
    chrom_seq = load_chrom_seq(loaded_chrom)

    for i, line in enumerate(f):
        if i % 1000 == 0: # ~ 357500 junctions
            print(f'Reading line {i}')
        line = line.split(',')
        junction = line[0].split('_')
        read_chrom = int(junction[0][-3:])
        start, end = int(junction[1]), int(junction[2])

        # if chromosome changes, update loaded sequence until chromosome 22 reached
        if read_chrom > loaded_chrom:
            if loaded_chrom == 22:
                break
            else:
                loaded_chrom += 1
                chrom_seq = load_chrom_seq(loaded_chrom)

        # extract sequence around start
        # todo: investigate why earliest start is 12058 eg for chrom 1
        # todo: i need to do this with respect to gene boundaries ;____;
            # however, not extremly big priorities as it essentially just some data pollution
        # q: does it work to just remove the first and last exon boundary found for each chromosome?
            # a: no, because that doesn't solve the problems for genes

        """ Filtering """
        # a minimal length of 25nt for the exons and a length of 80nt for the neighboring introns are
        # required as exons/introns shorter than 25nt/80nt are usually caused by sequencing errors and
        # they represent less than 1% of the exon and intron length distributions
        # https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6722613/
        if end - start < 25:
            continue
        gene_start = 0
        gene_end = 1e10

        # make sure there are at least 'introns_bef_start' intron nts between exon junctions
        # q: do i even want to do this? how do i do this?

        # if start - introns_bef_start < prev_end:
        #     continue

        # remove first exon in gene

        # remove last exon in gene
        if end + introns_after_end > gene_end:
            continue

        """ Extraction of the sequence """
        window_around_start = loaded_chrom[start-introns_bef_start:start+exons_after_start]
        window_around_end = loaded_chrom[end-exons_bef_end:end+introns_after_end]
        junction_seqs[line[0]] = [window_around_start, window_around_end]


# Write extracted sequences to file
with open(f'{data_path}/brain_cortex_junction_seqs.csv', 'w') as f:
    if i % 1000 == 0:  # ~ 357500 junctions
        print(f'Writing line {i}')
    for junction, reads in junction_seqs.items():
        f.write(f'{junction},{reads}\n')
print('Processing finished')
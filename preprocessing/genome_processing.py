# given junction reads from the correct samples, extract the corresponding intron-exon sequences from the genome
import csv
import linecache

data_path = '../data'
path_filtered_reads = f'{data_path}/brain_cortex_junction_reads.csv'

# todo do naive PSI estimation


introns_bef_start = 50 # introns
exons_after_start = 30 # exons

exons_bef_end = 30 # exons
introns_after_end = 50 # introns

# want to load chromosome as one giant string
def load_chrom_seq(chrom):
    with open(f'{data_path}/chr{chrom}.fa') as f:
        all = f.read().replace('\n', '')
        # reader = csv.reader(f, delimiter=" ")
        # lines = list(reader)
        # complete = ''.join(lines)
        # contains list with rows of samples
        return all

junction_seqs = {}
junction_psis = {}

with open(path_filtered_reads) as f:
    reader = csv.reader(f, delimiter="\n")
    # contains list with rows of samples
    all = list(reader)

with open(path_filtered_reads) as f:
    loaded_chrom = 1
    chrom_seq = load_chrom_seq(loaded_chrom)
    # all = list(f)
    print('Start of iteration through lines')
    for i, line in enumerate(f):
        if i % 1000 == 0: # ~ 357500 junctions
            print(f'Reading line {i}')
        # line = line.split(',')
        split_idx = line.find(',')
        junction = line[:split_idx].split('_')
        read_chrom = int(junction[0][3:])
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


        # make sure that very first exon in chromosone doesn't contain N nt input
        # make sure that very last exon in chromosone doesn't contain N nt input
        if i == 0 or i == len(all)-1: continue

        # if start - introns_bef_start < prev_end:
        #     continue

        # remove first exon in gene

        # remove last exon in gene
        if end + introns_after_end > gene_end:
            continue

        # """ Extraction of the sequence """
        window_around_start = chrom_seq[start-introns_bef_start:start+exons_after_start]
        window_around_end = chrom_seq[end-exons_bef_end:end+introns_after_end]
        junction_seqs[line[:split_idx]] = [window_around_start, window_around_end]

        """ Estimation of the PSI value """
        # PSI = pos / (pos + neg)
        first_read_idx = line[split_idx+1:].find(',')
        # yikes...
        pos = int(line[split_idx+2:split_idx+1+first_read_idx])
        # neg == it overlaps with the junction in any way:
        # one way to test for overlap:
        # (1) start2 <= start, end >= end2 >= start,
        # (2) end >= start2 >= start, end2 >= end
        # (3) end >= start2 >= start, end >= end2 >= start
        # (4) start2 <= start, end2 >= end

        # second way to test for overlap:
        # (1) end2 < start or
        # (2) start2 > end
        # => end2 >= start and start2 <= end
        # check all junctions above until end2 < start
        # problem: file too large into memory, but need specific lines from it
        # solution 1: save the last ~10 lines seen before
        # solution 2: realise that file only has 300 mb and it is fine to load it into memory
        neg = 0
        idx_above = i - 1
        # THIS MAKES THINGS SLOW AS FUCK ATM
        while True:
            if idx_above <= 0: break
            line2 = all[idx_above][0]
            break_idx = line2.find(',')
            junction2 = line2[:break_idx].split('_')
            start2, end2 = int(junction2[1]), int(junction2[2])
            if end2 < start: break
            idx_above -= 1
            kms = line2[break_idx+1:].find(',')
            # +1 for ','
            # +1 for '['
            neg += int(line2[break_idx+2:break_idx+kms+1])


        # check all junctions below until start2 > end
        idx_below = i+1
        while True:
            if idx_below >= len(all): break
            line2 = all[idx_below][0]
            break_idx = line2.find(',')
            junction2 = line2[:break_idx].split('_')
            start2, end2 = int(junction2[1]), int(junction2[2])
            if start2 > end: break
            idx_below += 1
            kms = line2[break_idx+1:].find(',')
            # +1 for ','
            # +1 for '['
            neg += int(line2[break_idx+2:break_idx+kms+1])
        if pos + neg == 0: psi = 0
        else: psi = pos / (pos + neg)
        junction_psis[line[:split_idx]] = psi


exit()


# Write extracted sequences to file
with open(f'{data_path}/brain_cortex_junction_seqs.csv', 'w') as f:
    print('Beginning to write extracted sequences')
    for junction, seqs in junction_seqs.items():
        f.write(f'{junction},{seqs}\n')

# Write estimated PSIs to file
with open(f'{data_path}/brain_cortex_junction_psis.csv', 'w') as f:
    print('Beginning to write estimated PSIs')
    for junction, psis in junction_psis.items():
        f.write(f'{junction},{psis}\n')

print('Processing finished')
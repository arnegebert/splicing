import csv

introns_bef_start = 70 # introns
exons_after_start = 70 # exons

exons_bef_end = 70 # exons
introns_after_end = 70 # introns
filtered = []
data_path = '../../data'
xxx = 1
src = 'hexevent/all_cassette_exons.txt'
target = 'hexevent/all_cassette_filtered_class.csv'
last_chrom = 22
gencode_idx = 1

# with open(f'../../data/gencode_exons.csv') as f:
#     print('xxxxxxxx')

# gencode = list(open(f'{data_path}/gencode_exons.csv').read().replace('\n', '').split('\t'))
gencode = {}
with open(f'{data_path}/gencode_exons.csv') as f:
    for line in f:
        line = line.split('\t')
        if len(line) == 1: continue
        chr, start, end = int(line[0][3:]), int(line[1]), int(line[2][:-1])
        if chr not in gencode:
            gencode[chr] = []
        gencode[chr].append((start, end))
    print('Finished reading gencode data')
    # reader = csv.reader(f, delimiter="\t")
    # contains list with rows of samples
    # gencode = list(reader)

print('Starting processing')
def load_chrom_seq(chrom):
    with open(f'{data_path}/chromosomes/chr{chrom}.fa') as f:
        all = f.read().replace('\n', '')
        # reader = csv.reader(f, delimiter=" ")
        # lines = list(reader)
        # complete = ''.join(lines)
        # contains list with rows of samples
        # log10 solution would be cleaner, but this is more readable
        # cutting out '>chr<chrom>'
        if chrom < 10:
            return all[5:]
        else:
            return all[6:]

def reverse_complement(seq):
    complt = []
    for bp in seq:
        if bp == 'a' or bp == 'A':
            complt.append('t')
        elif bp == 'c' or bp == 'C':
            complt.append('g')
        elif bp == 'g' or bp == 'G':
            complt.append('c')
        elif bp == 't' or bp == 'T':
            complt.append('a')
        else: raise Exception("Unidentified base-pair given")
    return ''.join(complt)

# todo add support for endings :|
def find_gencode_exon(read_chrom, start, end):
    gencode_idx_cpy = gencode_idx
    gencode_cpy = gencode[read_chrom]
    while True:
        gencode_start, gencode_end = gencode_cpy[gencode_idx_cpy]
        # gencode_start = int(gencode[gencode_idx_cpy][1])
        if gencode_start < start:
            gencode_idx_cpy += 1
        elif gencode_start == start:
            # gencode_end = int(gencode[gencode_idx_cpy][2])
            if gencode_end == end:
                # get next closest exon from left
                idx_left = gencode_idx_cpy - 1
                start_left = gencode_cpy[idx_left][0]
                while start_left == start:
                    idx_left -= 1
                    start_left = gencode_cpy[idx_left][0]
                end_left = gencode_cpy[idx_left][1]
                l1 = end_left - start

                # get next closest exon from right
                idx_right = gencode_idx_cpy + 1
                start_right = gencode_cpy[idx_right][0]
                while start_right == start:
                    idx_right += 1
                    start_right = gencode_cpy[idx_right][0]
                l3 = start_right - end
                return True, l1, l3, gencode_idx_cpy
        else:  # hopefully only happens rarely
            return False, None, None, gencode_idx_cpy

with open(f'{data_path}/{src}') as f:
    loaded_chrom = 1
    chrom_seq = load_chrom_seq(loaded_chrom)

    for i, line in enumerate(f):
        if i % 1000 == 0: # ~ 357500 junctions
            print(f'Reading line {i}')
        if i > 0:
            line = line.split('\t')
            # last line
            if len(line) == 1: continue
            count = int(line[4])
            start, end = int(line[2]), int(line[3])
            try:
                read_chrom = int(line[0][3:])
            except ValueError:
                break
            if read_chrom == last_chrom: break
            # if chromosome changes, update loaded sequence until chromosome 22 reached
            if read_chrom > loaded_chrom:
                loaded_chrom += 1
                chrom_seq = load_chrom_seq(loaded_chrom)

            """ Filtering """
            # min length of 25
            if end - start < 25: continue
            constit_level, skip = float(line[9]), int(line[8])

            # -------------------------------------------------------------------------------
            # low inclusion exons only
            # if constit_level > 0.2: continue
            # high inclusion exons only
            # if constit_level < 0.8: continue
            # -------------------------------------------------------------------------------


            # ~ 12280 cas exons, might make sense because skipping is the primary one
            if skip < 20 and count < 20: continue
            # if (skip < 20 or count < 4) and (count < 20 or skip < 4): continue

            # current state: this breaks in the first iteration;
            # TypeError: cannot unpack non-iterable NoneType object
            found, l1, l3, new_idx = find_gencode_exon(read_chrom, start, end)
            gencode_idx = new_idx
            if not found: continue

            # ~ 4564 cas exons
            # if count < 20: continue
            # if constit_level != 1 and skip < 4: continue

            # canonical splice sites = GT and AG
            # also existing: GC/AG and AT/AC
            window_around_start = chrom_seq[start - introns_bef_start - 2:start + exons_after_start]
            window_around_end = chrom_seq[end - exons_bef_end:end + introns_after_end + 2]
            strand = line[1]
            # if strand == '+':
            #     # almost always AG
            #     # print(chrom_seq[start-2:start])
            #     # almost always GT
            #     print(chrom_seq[end-0:end+2])
            if strand == '-':
                window_around_start = reverse_complement(window_around_start)
                window_around_end = reverse_complement(window_around_end)
            # always the case since constitutive exons only atm
            is_constitutive = 0.0

            # if float(line[9]) != 1: print(f'{i}: no')
            junction = f'{line[0]}_{start}_{end}'
            filtered.append(
                (junction, window_around_start, window_around_end, is_constitutive)
            )

print(len(filtered))
with open(f'{data_path}/{target}', 'w') as f:
    for (junction, start, end, psi) in filtered:
        f.write(f'{junction}\t{start}\t{end}\t{psi}\n')
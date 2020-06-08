import csv
import numpy as np

srcs = ['hexevent/all_cons.txt', 'hexevent/all_cass.txt', 'hexevent/all_cass.txt', 'hexevent/all_cass.txt']
targets = ['hexevent/all_cons_filtered_class.csv', 'hexevent/all_cass_filtered_class.csv',
           'hexevent/low_cass_filtered_class.csv', 'hexevent/high_cass_filtered_class.csv']
settings = [(True, False, False), (False, False, False), (False, True, False), (False, False, True)]

# gencode = list(open(f'{data_path}/gencode_exons.csv').read().replace('\n', '').split('\t'))
gencode = {}
with open(f'../../data/gencode_exons.csv') as f:
    for line in f:
        line = line.split('\t')
        if len(line) == 1: continue
        chr, start, end = int(line[0][3:]), int(line[1]), int(line[2][:-1])
        if chr not in gencode:
            gencode[chr] = []
        gencode[chr].append((start, end))
    print('Finished reading gencode data')

for src, target, (cons, low, high) in zip(srcs, targets, settings):

    introns_bef_start = 70 # introns
    exons_after_start = 70 # exons

    exons_bef_end = 70 # exons
    introns_after_end = 70 # introns
    # whether to use the same mean and std among all datasets or compute a per dataset one
    use_universal_normalization = True
    data_path = '../../data'
    # src = 'hexevent/all_cons.txt'
    # target = 'hexevent/all_cons_filtered_class.csv'
    # # src = 'hexevent/all_cass.txt'
    # # target = 'hexevent/all_cass_filtered_class.csv'
    #
    # cons = True
    # low = False
    # high = False


    assert not(low and high), 'filter either for low or high inclusion rate'
    assert not(cons and (low or high)), 'can\'t filter according to inclusion level and be constitutive at the same time'

    filtered = []
    last_chrom = 22
    gencode_idx = 0

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

    not_annotated = 0
    too_short = 0
    # O(N) search for intron & exon lengths :) smiley-face
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
                if gencode_end < end:
                    gencode_idx_cpy += 1
                elif gencode_end > end:
                    return False, None, None, gencode_idx_cpy
                elif gencode_end == end:
                    # get next closest exon from left
                    idx_left = gencode_idx_cpy - 1
                    start_left, end_left = gencode_cpy[idx_left]
                    while end_left >= start:
                        idx_left -= 1
                        start_left, end_left = gencode_cpy[idx_left]
                    l1 = start - end_left
                    if l1 == 0: print('It happened for l1')

                    # get next closest exon from right
                    idx_right = gencode_idx_cpy + 1
                    start_right = gencode_cpy[idx_right][0]
                    while start_right <= end:
                        idx_right += 1
                        start_right = gencode_cpy[idx_right][0]
                    l3 = start_right - end
                    if l3 == 0: print('It happened for l3')
                    return True, l1, l3, gencode_idx_cpy
            else:  # hopefully only happens rarely
                # print(f'Left like this on iteration {i}')
                return False, None, None, gencode_idx_cpy

    exon_lens = []
    intron_lens = []
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
                    gencode_idx = 0
                    loaded_chrom += 1
                    chrom_seq = load_chrom_seq(loaded_chrom)

                """ Filtering """

                constit_level, skip = float(line[9]), int(line[8])

                # -------------------------------------------------------------------------------
                # low inclusion exons only
                if low and constit_level > 0.2: continue
                # high inclusion exons only
                if high and constit_level < 0.8: continue
                # -------------------------------------------------------------------------------


                # ~ 12280 cas exons, might make sense because skipping is the primary one
                if skip < 20 and count < 20: continue
                # if (skip < 20 or count < 4) and (count < 20 or skip < 4): continue

                # extracting intron length and
                found, l1, l3, new_idx = find_gencode_exon(read_chrom, start, end)
                gencode_idx = new_idx
                if not found:
                    not_annotated += 1
                    continue
                if l1 < 80 or l3 < 80:
                    too_short += 1
                    continue
                # min length of 25
                l2 = end - start
                if end - start < 25: continue
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
                if cons:
                    is_constitutive = 1.0
                else:
                    is_constitutive = 0.0

                # if float(line[9]) != 1: print(f'{i}: no')
                junction = f'{line[0]}_{start}_{end}'
                filtered.append(
                    (junction, window_around_start, window_around_end, is_constitutive, l1, l2, l3)
                )
                exon_lens.append(l2)
                intron_lens.append(l1)
                intron_lens.append(l3)

    print(f'Number of exons after filtering: {len(filtered)}')
    print(f'Number of skipped not-annotated exons: {not_annotated} ')
    print(f'Number of skipped exons with too short neighbouring introns l1 or l3: {too_short}')
    exon_lens = np.array(exon_lens)
    intron_lens = np.array(intron_lens)
    exon_mean, exon_std = np.mean(exon_lens), np.std(exon_lens)
    intron_mean, intron_std = np.mean(intron_lens), np.std(intron_lens)
    print(f'Exon mean: {exon_mean}')
    print(f'Exon std: {exon_std}')
    print(f'Intron mean: {intron_mean}')
    print(f'Intron std: {intron_std}')

    # Exon mean: 121.46372945387063
    # Exon std: 77.96368240100577
    # Intron mean: 3513.0288308589606
    # Intron std: 8309.354849020463
    if use_universal_normalization:
        exon_mean, exon_std = 121.46372945387063, 77.96368240100577
        intron_mean, intron_std = 3513.0288308589606, 8309.354849020463
    with open(f'{data_path}/{target}', 'w') as f:
        for (junction, start, end, psi, l1, l2, l3) in filtered:
            f.write(f'{junction}\t{start}\t{end}\t{psi}\t{(l1-intron_mean)/intron_std}\t'
                    f'{(l2-exon_mean)/exon_std}\t{(l3-intron_mean)/intron_std}\n')
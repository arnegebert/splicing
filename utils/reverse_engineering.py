import numpy as np
from timeit import default_timer as timer

def load_start_of_legit_sequences():
    startt = timer()
    gencode = {}
    with open(f'../data/gencode_exons.csv') as f:
        for line in f:
            line = line.split('\t')
            if len(line) == 1: continue
            chr, start, end = int(line[0][3:]), int(line[1]), int(line[2][:-1])
            if chr not in gencode:
                gencode[chr] = []
            gencode[chr].append((start, end))
        print('Finished reading gencode data')

    src = 'send_to_me_DSC/all_cons.txt'
    cons, low, high = (True, False, False)
    introns_bef_start = 70 # introns
    exons_after_start = 70 # exons

    exons_bef_end = 70 # exons
    introns_after_end = 70 # introns
    # whether to use the same mean and std among all datasets or compute a per dataset one
    use_universal_normalization = True
    data_path = '../data'
    assert not(low and high), 'filter either for low or high inclusion rate'
    assert not(cons and (low or high)), 'can\'t filter according to inclusion level and be constitutive at the same time'

    filtered = []
    last_chrom = 23
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
                # found, l1, l3, new_idx = find_gencode_exon(read_chrom, start, end)
                # gencode_idx = new_idx
                # if not found:
                #     not_annotated += 1
                #     continue
                # if l1 < 80 or l3 < 80:
                #     too_short += 1
                #     continue
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
                # filtered.append(
                #     # (junction, window_around_start, window_around_end, float(line[9]), l1, l2, l3)
                #     (junction, window_around_start, window_around_end, is_constitutive)
                # )
                filtered.append(window_around_start)

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

    endt = timer()
    print(f'It took {endt-startt} s to generate the data sets')
    return filtered

def one_hot_decode(nt):
    if (nt == [1.0, 0, 0, 0]).all():
        return 'A'
    elif (nt == [0, 1.0, 0, 0]).all():
        return 'C'
    elif (nt == [0, 0, 1.0, 0]).all():
        return 'G'
    elif (nt == [0, 0, 0, 1.0]).all():
        return 'T'

def decode_seq_vanilla(seq):
    to_return = []
    for encoding in seq:
        to_return.append(one_hot_decode(encoding))
    return to_return

def decode_seq(seq, dct):
    to_return = []
    for encoding in seq:
        to_return.append(dct[encoding])
    return to_return

def build_dictionary_off_permutation(p):
    dct = {}
    encodings = [np.array([1.0, 0, 0, 0]), np.array([0, 1.0, 0, 0]), np.array([0, 0, 1.0, 0]), np.array([0, 0, 0, 1.0])]
    encodings = [[1.0, 0, 0, 0], [0, 1.0, 0, 0], [0, 0, 1.0, 0], [0, 0, 0, 1.0]]
    encodings = [(1.0, 0, 0, 0), (0, 1.0, 0, 0), (0, 0, 1.0, 0), (0, 0, 0, 1.0)]
    for nt, enc in zip(p, encodings):
        dct[enc] = nt
    return dct

def generate_all_permutations(string):
    if not string:
        return ['']
    perms = []
    for c in string:
        sub_perms = generate_all_permutations(string.replace(c, ''))
        for sub_perm in sub_perms:
            perms.append(c + sub_perm)
    return perms


x_cons_data = np.load('../data/hexevent/x_cons_data.npy')
legit_sequences = load_start_of_legit_sequences()
perms = generate_all_permutations('ACGT')
shiny_pokemon = 0

for seq_idx in range(30):
    print(f'Testing sample with id {seq_idx}')
    print('--------------------------------')
    tester = x_cons_data[seq_idx, :140, :4]
    xxx = []
    for enc in tester:
        xxx.append(tuple(enc))
    tester = xxx
    # decoded = decode_seq_vanilla(tester)
    # decoded = ''.join(decoded)
    # print('plis')

    # legit_sequences = ['A']


    for i, p in enumerate(perms):
        print(f'Testing permutation {i}: {p}')
        dct = build_dictionary_off_permutation(p)
        decoded = decode_seq(tester, dct)
        decoded = ''.join(decoded)
        extract = decoded#[5:65]
        for seq in legit_sequences:
            if extract in seq:
                shiny_pokemon += 1
                print('---------------------------------------------------------')
                print('XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX')
                print('XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX')
                print('XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX')
                print('XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX')


print(f'Found {shiny_pokemon} shiny pokemon')
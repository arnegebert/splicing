import numpy as np
import csv
import gensim.models
import time

startt = time.time()
data_path_decoded = '../data/distributed'
data_path_origin = '../data/send_to_me_DSC'
# data_path_origin = '../data/hexevent'

lifehack = 100000
# decoded data
with open(f'{data_path_decoded}/decoded_cons_data_class.csv') as f:
    reader = csv.reader(f, delimiter='\t')
    decoded_cons = list(reader)
decoded_cons = np.array(decoded_cons)[:lifehack]

with open(f'{data_path_decoded}/decoded_cas_data_high_class.csv') as f:
    reader = csv.reader(f, delimiter='\t')
    hx_cas_data = list(reader)
hx_cas_data = np.array(hx_cas_data)[:lifehack]

with open(f'{data_path_decoded}/decoded_cas_data_low_class.csv') as f:
    reader = csv.reader(f, delimiter='\t')
    lx_cas_data = list(reader)
lx_cas_data = np.array(lx_cas_data)[:lifehack]
decoded_cas = np.concatenate((hx_cas_data, lx_cas_data))

# original exons
with open(f'{data_path_origin}/all_cons.txt') as f:
    reader = csv.reader(f, delimiter='\t')
    original_cons = list(reader)
with open(f'{data_path_origin}/all_cass.txt') as f:
    reader = csv.reader(f, delimiter='\t')
    original_cass = list(reader)



last_chrom = 23
introns_bef_start = 70 - 1 # introns
exons_after_start = 70 + 1 # exons

exons_bef_end = 70 -2 # exons
introns_after_end = 70 # introns

def load_chrom_seq(chrom):
    with open(f'../data/chromosomes/chr{chrom}.fa') as f:
        loaded_chrom_seq = f.read().replace('\n', '')
        if chrom < 10:
            return loaded_chrom_seq[5:]
        else:
            return loaded_chrom_seq[6:]

def reverse_complement(seq):
    complt = []
    for bp in seq:
        if bp == 'a' or bp == 'A':
            complt.append('T')
        elif bp == 'c' or bp == 'C':
            complt.append('G')
        elif bp == 'g' or bp == 'G':
            complt.append('C')
        elif bp == 't' or bp == 'T':
            complt.append('A')
        else: raise Exception("Unidentified base-pair given")
    return ''.join(complt)

def get_used_exons(decoded, original):
    pos, neg = 0, 0
    used_exons = []
    loaded_chrom = 1
    chrom_seq = load_chrom_seq(loaded_chrom)
    hashs = set()
    for seq in decoded[:, 0]:
        hashs.add(seq)

    print('Start of iteration through lines')
    for i, line in enumerate(original):
        if i == 0: continue
        # if i % 1000 == 0:
        #     print(f'Reading line {i}')
        chrom, strand, start, end = line[0], line[1], int(line[2]), int(line[3])
        count, skip, constit_level = int(line[4]), int(line[8]), float(line[9])
        try:
            read_chrom = int(line[0][3:])
        except ValueError:
            break
        if read_chrom == last_chrom: break
        # if chromosome changes, update loaded sequence until chromosome 22 reached
        if read_chrom > loaded_chrom:
            loaded_chrom += 1
            current_idx_overlap = 0
            chrom_seq = load_chrom_seq(loaded_chrom)

        window_around_start = chrom_seq[start - introns_bef_start - 1:start + exons_after_start - 1].upper()
        window_around_end = chrom_seq[end - exons_bef_end - 2:end + introns_after_end - 2].upper()
        if strand == '+':
            pos += 1
        if strand == '-':
            neg += 1
            # window_around_start = window_around_start[73:130]
            window_around_start = window_around_start[::-1]
            window_around_start = reverse_complement(window_around_start)
            window_around_end = window_around_end[5:-5:-1]

        # in_there = any([window_around_start in seq for seq in decoded[:, 0]])
        in_there = window_around_start in hashs
        # if i in [20286]:
        #     test = set()
        #     test.add(decoded[1,0])
        #     test2 = set(window_around_start)
        #     test3 = dict()
        #     print(window_around_start==decoded[1,0])
        #     print(window_around_start in test)
        #     print(decoded[1,0] in test2)
        #     print('meh')
        if in_there:
            used_exons.append((chrom, strand, start, end, count, skip, constit_level))
            print(i)
            # in_there2 = any([window_around_end in seq for seq in decoded[:, 1]])
            # if in_there2: print('yep')
    print(pos)
    print(neg)
    return used_exons

# wowww -- 200-times speed up through hashing :>
# original: 64902/65262 +/- cons exons
# original: 28827/28116 +/- neg exons
# 19569/39128 + cons
# 5876/11790 + cass
# finding 3506 from - strand of 39128 total cons exons
# finding 1552 from - strand out of 11790 total exons
print('Starting processing cons exons')
used_cons_exons = get_used_exons(decoded_cons, original_cons)
print(f'Found {len(used_cons_exons)} matching constitutive exons')
print(f'Wanted to find {len(decoded_cons)} constitutive exons')
with open('../data/partial_junction/cons_exons.csv', 'w') as f:
    for chrom, strand, start, end, count, skip, constit_level in used_cons_exons:
        f.write(f'{chrom}\t{strand}\t{start}\t{end}\t{count}\t{skip}\t{constit_level}\n')


print('Starting processing cass exons')
used_cass_exons = get_used_exons(decoded_cas, original_cass)
print(f'Found {len(used_cass_exons)} matching cassette exons')
print(f'Wanted to find {len(decoded_cas)} cassette exons')
with open('../data/partial_junction/cass_exons.csv', 'w') as f:
    for chrom, strand, start, end, count, skip, constit_level in used_cass_exons:
        f.write(f'{chrom}\t{strand}\t{start}\t{end}\t{count}\t{skip}\t{constit_level}\n')

endt = time.time()
print(f'Time to process data: {endt-startt}')
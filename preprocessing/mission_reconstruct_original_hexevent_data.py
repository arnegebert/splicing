import numpy as np
import csv
import gensim.models
import time

startt = time.time()
data_path_decoded = '../data/distributed'
data_path_origin = '../data/send_to_me_DSC'
# data_path_origin = '../data/hexevent'
lifehack = 1000000

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
introns_bef_start = 70# introns
exons_after_start = 70# exons

exons_bef_end = 70 # exons
introns_after_end = 70# introns

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
    hasht = dict()
    for i, seq in enumerate(decoded[:, 0]):
        hasht[seq] = i

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

        window_around_start = chrom_seq[start - introns_bef_start:start + exons_after_start].upper()
        window_around_end = chrom_seq[end - exons_bef_end:end + introns_after_end].upper()
        if strand == '+':
            pos += 1
            in_there = window_around_start in hasht
        if strand == '-':
            continue
            neg += 1
            # window_around_start = window_around_start[70:]
            cpy = window_around_start[105:]
            cpy = cpy[::-1]
            cpy = reverse_complement(cpy)
            # window_around_start = window_around_start[::-1]
            # window_around_start = reverse_complement(window_around_start)
            in_there = any([cpy in seq for seq in decoded[:, 0]])
            # match beginning at index 62
            # 31 the other time
            # before 62, also matches sequence before
            # therefore, extracting moved by 62 nts + reversed + reverse complemented????
        if in_there: # +: 20286, -: 42079
            decoded_idx = hasht[window_around_start]
            l1, l2, l3 = decoded[decoded_idx, 2:5]
            used_exons.append((chrom, strand, start, end, count, skip, constit_level, l1, l2, l3))
            print(i)
            # print(f'Offset index: {decoded[[cpy in seq for seq in decoded[:, 0]].index(True),0].find(cpy)}')
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
with open('../data/dsc_reconstruction_junction/cons_exons.csv', 'w') as f:
    for chrom, strand, start, end, count, skip, constit_level, l1, l2, l3 in used_cons_exons:
        f.write(f'{chrom}\t{strand}\t{start}\t{end}\t{count}\t{skip}\t{constit_level}\t{l1}\t{l2}\t{l3}\n')


print('Starting processing cass exons')
used_cass_exons = get_used_exons(decoded_cas, original_cass)

with open('../data/dsc_reconstruction_junction/cass_exons.csv', 'w') as f:
    for chrom, strand, start, end, count, skip, constit_level, l1, l2, l3 in used_cass_exons:
        f.write(f'{chrom}\t{strand}\t{start}\t{end}\t{count}\t{skip}\t{constit_level}\t{l1}\t{l2}\t{l3}\n')

print(f'Found {len(used_cons_exons)} matching constitutive exons')
print(f'Wanted to find {len(decoded_cons)} constitutive exons')
endt = time.time()
print(f'Time to process data: {endt-startt}')

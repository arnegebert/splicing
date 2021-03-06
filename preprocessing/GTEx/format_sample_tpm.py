import csv
import numpy as np
import argparse
import time

parser = argparse.ArgumentParser()
parser.add_argument('--tissue', type=str, default='Brain - Cortex', metavar='tissue',
                    help='type of tissue filtered for')
parser.add_argument('--save-to', type=str,
                    default='../../data/gtex_processed/brain_cortex_junction_reads_one_sample.csv',
                    metavar='save_to', help='path to folder and file to save to')
args = parser.parse_args()


path_annotation = '../../data/gtex_origin/GTEx_Analysis_v8_Annotations_SampleAttributesDS.txt'
path_srcs = '../../data/gtex_origin/GTEx_Analysis_2017-06-05_v8_RNASeQCv1.1.9_gene_tpm.gct'
# path_srcs = '../data/gtex/GTEx_Analysis_2017-06-05_v8_RSEMv1.3.0_transcript_tpm.gct'
target = '../../data/gtex_processed/brain_cortex_tpm_one_sample.csv'
include_version = False

startt = time.time()
with open(path_annotation) as f:
    reader = csv.reader(f, delimiter="\t")
    # contains list with rows of samples
    d = list(reader)

tissue = args.tissue
target = args.save_to
#'Brain - Cortex'

def samples_from_tissue(data, tissue):
    filtered_sample_names = []
    for row in data:
        if tissue in row:
            filtered_sample_names.append(row[0])
    return filtered_sample_names

filtered_sample_names = samples_from_tissue(d, tissue)
# first 5 for 'Brain - Cortex':
# GTEX-1117F-3226-SM-5N9CT
# GTEX-111FC-3126-SM-5GZZ2
# GTEX-1128S-2726-SM-5H12C
# GTEX-117XS-3026-SM-5N9CA
# GTEX-117XS-3026-SM-CYKOR
# for i in range(5):
#     print(filtered_sample_names[i])
print(f'{len(filtered_sample_names)} samples after getting those with {tissue} data')

# next step:
# given the sample names, preprocess junction data file such that it only contains junctions from desired samples
# have to work with that very large python text file as a result
gtex_sample_id = 164
sample_idxs = []
data = {}
with open(path_srcs) as f:
    for i, line in enumerate(f):
        if i % 1000 == 0: # ~ 357500 junctions
            print(f'Reading line {i}')
        line = line.split('\t')
        if i == 2:
            for n in filtered_sample_names:
                try:
                    sample_idxs.append(line.index(n))
                except ValueError:
                    # print(f'{n} not in .gtc-file')
                    pass
            print(f'{len(sample_idxs)} samples after only using those in .gtc-file')
        elif i > 2:
            # data_line = []
            # for idx in sample_idxs:
            #     data_line.append(line[idx])
            # data[line[0]] = '\t'.join(data_line)
            gene = line[0]
            if not include_version:
                version_idx = gene.index('.')
                gene = gene[:version_idx]
            tpm = float(line[sample_idxs[gtex_sample_id]])
            data[gene] = tpm
# data = junctions -> sample reads

# todo: some junctions are duplicated in the data... o.o
# 352051 are only in the dictionary

# planning:
# in what format do i want data to later load it?
# at first, junction : list of reads might not be bad
# later i will need to extract the information from the junctions either way
print(f'{len(data)} genes after filtering')
with open(target, 'w') as f:
    print('Beginning to write gene TPMs')
    for gene, tpms in data.items():
        f.write(f'{gene},{tpms}\n')

endt = time.time()
print(f'Processing finished in {endt-startt:.2f} s')

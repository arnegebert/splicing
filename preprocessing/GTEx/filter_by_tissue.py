import csv
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--tissue', type=str, default='Brain - Cortex', metavar='tissue',
                    help='type of tissue filtered for')
parser.add_argument('--save-to', type=str,
                    default='../../data/gtex_processed/brain_cortex_junction_reads_one_sample.csv',
                    metavar='save_to', help='path to folder and file to save to')
args = parser.parse_args()

tissues = ['Brain - Cortex', 'Brain - Cerebellum', 'Heart - Left Ventricle']
tissue = args.tissue
# tissue = 'Heart - Left Ventricle'# args.tissue
# tissue = 'Brain - Cerebellum'
# Brain - Cortex == Brain - Frontal Cortex (BA9)
# Brain - Cerebellum == Brain - Cerebellar Hemisphere
assert tissue in ['Brain - Cortex', 'Brain - Cerebellum', 'Heart - Left Ventricle']
path_annotation = '../../data/gtex_origin/GTEx_Analysis_v8_Annotations_SampleAttributesDS.txt'
path_srcs = '../../data/gtex_origin/GTEx_Analysis_2017-06-05_v8_STARv2.5.3a_junctions.gct'

save_to = args.save_to
# save_to = '../../data/gtex_processed/brain_cortex_junction_reads_one_sample.csv'
# path_srcs = '../../data/gtex_origin/GTEx_Analysis_2017-06-05_v8_RNASeQCv1.1.9_gene_tpm.gct'

# Brain - Cortex sample with id 164 corresonds to sample id 'GTEX-1F7RK-1826-SM-7RHI4'
# that is, 1F7RK is the donor id
G = 6
H = 7
with open(path_annotation) as f:
    reader = csv.reader(f, delimiter="\t")
    # contains list with rows of samples
    d = list(reader)

def samples_from_tissue(data, tissue):
    filtered_sample_names = []
    for row in data:

        if tissue in row:
            filtered_sample_names.append(row[0])
    return filtered_sample_names


def samples_from_tissues(data, tissues):
    filtered_sample_names = []
    for tissue in tissues:
        filtered_sample_names.append(samples_from_tissue(data, tissue))
    return filtered_sample_names

filtered_sample_names = samples_from_tissue(d, tissue)
desired_donor = '1HCVE'
desired_donor_matches = [desired_donor in name for name in filtered_sample_names]
idx_desired_sample = desired_donor_matches.index(True)

if sum(desired_donor_matches) > 1:
    print('-' * 40)
    print('Warning: ')
    print('There are multiple samples of the selected tissue type from the desired donor in the database')
    print(f'Proceeding with the first one found at index: {idx_desired_sample}')
    print('-' * 40)

# first 5 for 'Brain - Cortex':
# GTEX-1117F-3226-SM-5N9CT
# GTEX-111FC-3126-SM-5GZZ2
# GTEX-1128S-2726-SM-5H12C
# GTEX-117XS-3026-SM-5N9CA
# GTEX-117XS-3026-SM-CYKOR
# for i in range(5):
#     print(filtered_sample_names[i])
print(f'{len(filtered_sample_names)} samples after getting those with brain cortex data')

# next step:
# given the sample names, preprocess junction data file such that it only contains junctions from desired samples
# have to work with that very large python text file as a result
sample_idxs = []
data = []
total_reads = 0
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
            gene = line[1]
            dot_idx = gene.index('.')
            gene = gene[:dot_idx]
            junction = line[0]
            reads = int(line[sample_idxs[idx_desired_sample]])
            total_reads += reads
            data.append((gene, junction, reads))
# data = junctions -> sample reads

print(f'Selected tissue sample has {total_reads} total reads')
# todo: some junctions are duplicated in the data... o.o
# 352051 are only in the dictionary

# planning:
# in what format do i want data to later load it?
# at first, junction : list of reads might not be bad
# later i will need to extract the information from the junctions either way
with open(save_to, 'w') as f:
    print('Beginning to write junction reads')
    for (gene, strand, junction, reads) in data:
        f.write(f'{junction},{strand},{reads},{gene}\n')
print('Processing finished')

import csv
import numpy as np

path_annotation = '../data/GTEx_Analysis_v8_Annotations_SampleAttributesDS.txt'
path_junctions = '../data/GTEx_Analysis_2017-06-05_v8_STARv2.5.3a_junctions.gct'
# d = np.loadtxt(path, delimiter='\t')

G = 6
H = 7
with open(path_annotation) as f:
    reader = csv.reader(f, delimiter="\t")
    # contains list with rows of samples
    d = list(reader)

tissue = 'Brain - Cortex'

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
print(f'{len(filtered_sample_names)} samples after getting those with brain cortex data')

# next step:
# given the sample names, preprocess junction data file such that it only contains junctions from desired samples
# have to work with that very large python text file as a result
sample_idxs = []
data = {}
with open(path_junctions) as f:
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
            data_line = []
            for idx in sample_idxs:
                data_line.append(line[idx])
            data[line[0]] = '\t'.join(data_line)
            # data[line[0]] = int(line[sample_idxs[164]])
# data = junctions -> sample reads

# todo: some junctions are duplicated in the data... o.o
# 352051 are only in the dictionary

# planning:
# in what format do i want data to later load it?
# at first, junction : list of reads might not be bad
# later i will need to extract the information from the junctions either way
with open('../data/brain_cortex_junction_reads.csv', 'w') as f:
    print('Beginning to write junction reads')
    for junction, reads in data.items():
        f.write(f'{junction},{reads}\n')
print('Processing finished')
# print(data)

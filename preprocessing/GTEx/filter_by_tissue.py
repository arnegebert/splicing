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

tissue = args.tissue
# tissue = 'Heart - Left Ventricle'# args.tissue
# tissue = 'Brain - Cerebellum'
# Brain - Cortex == Brain - Frontal Cortex (BA9)
# Brain - Cerebellum == Brain - Cerebellar Hemisphere
allowed_tissues = ['Brain - Cortex', 'Brain - Cerebellum', 'Heart - Left Ventricle']
assert tissue in allowed_tissues
path_annotation = '../../data/gtex_origin/GTEx_Analysis_v8_Annotations_SampleAttributesDS.txt'
path_junction_reads = '../../data/gtex_origin/GTEx_Analysis_2017-06-05_v8_STARv2.5.3a_junctions.gct'

save_to = args.save_to
# save_to = '../../data/gtex_processed/brain_cortex_junction_reads_one_sample.csv'
# path_srcs = '../../data/gtex_origin/GTEx_Analysis_2017-06-05_v8_RNASeQCv1.1.9_gene_tpm.gct'

# Brain - Cortex sample with id 164 corresonds to sample id 'GTEX-1F7RK-1826-SM-7RHI4'
# that is, 1F7RK is the donor id

def load_annotation_file(path_annotation):
    with open(path_annotation) as f:
        reader = csv.reader(f, delimiter="\t")
        # contains list with rows of samples
        d = list(reader)
    return d

annotation_data = load_annotation_file(path_annotation)

def find_sample_with_desired_donor_and_tissue_type(annotation_data, tissue, donor_name):
    for i, row in enumerate(annotation_data):
        curr_sample_id = row[0]
        if donor_name in curr_sample_id: # found sample from desired donor
            if tissue in row: # sample from donor of sought after tissue type
                return curr_sample_id

# This donor has tissue samples from all the tissues we want to examine
# Was selected in a previous processing step
desired_donor = '1HCVE'
target_sample_name = find_sample_with_desired_donor_and_tissue_type(annotation_data, tissue, desired_donor)

sample_idxs = []
data = []
total_reads = 0
with open(path_junction_reads) as f:
    for i, line in enumerate(f):
        if i % 1000 == 0: # ~ 357500 junctions
            print(f'Reading line {i}')
        line = line.split('\t')
        if i < 2: continue
        elif i == 2: # line 2 contains all sample names
            desired_donor_idx = line.find(target_sample_name)
        elif i > 2:
            gene = line[1]
            dot_idx = gene.index('.')
            gene = gene[:dot_idx]
            junction = line[0]
            reads = int(line[desired_donor_idx])
            total_reads += reads
            data.append((gene, junction, reads))

print(f'Selected tissue sample has {total_reads} total reads')

with open(save_to, 'w') as f:
    print('Beginning to write junction reads')
    for (gene, junction, reads) in data:
        f.write(f'{junction},{reads},{gene}\n')
print('Processing finished')

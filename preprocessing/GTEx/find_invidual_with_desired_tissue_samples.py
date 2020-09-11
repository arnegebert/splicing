import csv

# helper script to filter for donor who has tissue samples from brain cortex, cerebellum and heart

path_annotation = '../../data/gtex_origin/GTEx_Analysis_v8_Annotations_SampleAttributesDS.txt'
tissues = ['Brain - Cortex', 'Brain - Cerebellum', 'Heart - Left Ventricle']
with open(path_annotation) as f:
    reader = csv.reader(f, delimiter="\t")
    # contains list with rows of samples
    d = list(reader)

def samples_from_tissue(data, tissue):
    filtered_sample_names = []
    for row in data:
        if tissue in row:
            sample_id = row[0].split('-')
            donor_id = sample_id[1]
            filtered_sample_names.append(donor_id)
    return filtered_sample_names

first = True
seen = []
for tissue in tissues:
    filtered_sample_names = set(samples_from_tissue(d, tissue))
    if first:
        seen = list(filtered_sample_names)
        first = False
    else:
        seen = [sample for sample in seen if sample in filtered_sample_names]

print(f'Number of samples left: {len(seen)}')
print(seen)
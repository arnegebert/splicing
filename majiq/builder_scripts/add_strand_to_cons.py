from utils import timer

geneid_to_juncs = dict()
builder_dir = '../builder_not_neuron'

@timer
def build_hash_table(builder_dir=builder_dir):
    with open(f'{builder_dir}/cons_junc_sorted.tsv') as f:
        for i, line in enumerate(f):
            if i % 1000 == 0: print(f'Line {i}')
            # build hash table geneid -> juncs
            line = line.replace('\n','')
            geneid, chrom, jstart, jend, dstart, dend, astrat, aend = line.split('\t')
            geneid = geneid[5:]
            if geneid not in geneid_to_juncs:
                geneid_to_juncs[geneid] = []
            geneid_to_juncs[geneid].append(line)

@timer
def iterate_through_annotation_file():
    double_annotated = 0
    with open('../../data/gencode.v34.annotation.gtf') as f:
        for i, line in enumerate(f):
            if i % 10000 == 0: print(f'Line {i}')
            line = line.split('\t')
            if len(line) < 2: continue
            biotype = line[2]
            if biotype == 'gene':
                comments = line[8].split(';')
                geneid = comments[0]
                upto = geneid.find('.')
                # removing 'gene_id "' and versioning
                geneid = geneid[9:upto]
                if geneid in geneid_to_juncs:
                    if line[6] in ['+', '-']: chrom = line[6]
                    elif line[7] in ['+', '-']: chrom = line[7]
                    else: raise Exception('No strand information found')
                    # pretty ugly
                    target = geneid_to_juncs[geneid]
                    if target[-1] not in ['+', '-']:
                        target.append(chrom)
                    else: double_annotated += 1
    print(f'Number of double annotations: {double_annotated}')

@timer
def write_result(builder_dir=builder_dir):
    unmatched_geneids = 0
    with open(f'{builder_dir}/cons_junc_sorted_stranded.tsv', 'w') as f:
        f.write(f'#GENEID\tCHROMOSOME\tJUNC_START\tJUNC_END\tDONOR_START\tDONOR_END\tACCEPTOR_START\tACCEPTOR_END\tSTRAND\n')
        for geneid in geneid_to_juncs.keys():
            values = geneid_to_juncs[geneid]
            strand = values[-1]
            if strand in ['+', '-']:
                for i in range(len(values)-1):
                    f.write(values[i]+ '\t' + strand+'\n')
            else: unmatched_geneids += 1
    print(f'Number of unmatched geneids: {unmatched_geneids}')

build_hash_table()
iterate_through_annotation_file()
write_result()
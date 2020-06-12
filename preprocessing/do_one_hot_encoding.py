import pickle

src = '../data/gtex_processed/brain_cortex_full.csv'
target = '../data/brain_cortex_full.csv'


def one_hot_encode(nt):
    if nt == 'A' or nt == 'a':
        return [1.0, 0, 0, 0]
    elif nt == 'C' or nt == 'c':
        return [0, 1.0, 0, 0]
    elif nt == 'G' or nt == 'g':
        return [0, 0, 1.0, 0]
    elif nt == 'T' or nt == 't':
        return [0, 0, 0, 1.0]

def encode_seq(seq):
    encoding = []
    for nt in seq:
        encoding.append(one_hot_encode(nt))
    return encoding


samples = []
with open(src, 'r') as f:
    for i, l in enumerate(f):
        j, start_seq, end_seq, psi = l.split(',')
        psi = float(psi[:-1])
        sample = (encode_seq(start_seq), encode_seq(end_seq), psi)
        samples.append(sample)

# with open(target, 'w') as f:
#     print('Beginning to write estimated PSIs and extracted encoded sequences')
#     for (junction, start_seq, end_seq, psi) in samples:
#         f.write(f'{junction}\t{start_seq}\t{end_seq}\t{psi}\n')

with open(target, 'wb') as f:
    print('Beginning to write estimated PSIs and extracted encoded sequences')
    pickle.dump(samples, f)
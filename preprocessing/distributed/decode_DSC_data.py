import numpy as np

def one_hot_decode(nt):
    if (nt == [1.0, 0, 0, 0]).all():
        return 'A'
    elif (nt == [0, 1.0, 0, 0]).all():
        return 'C'
    elif (nt == [0, 0, 1.0, 0]).all():
        return 'G'
    elif (nt == [0, 0, 0, 1.0]).all():
        return 'T'
    else: raise Exception('Unknown encoding. Decoding failed. ')

def decode_batch_of_seqs(batch):
    to_return = []
    for seq in batch:
        temp = []
        for encoding in seq:
            temp.append(one_hot_decode(encoding))
        to_return.append(''.join(temp))
    return to_return


# Constant values taken in reference from
# https://github.com/louadi/DSC/blob/master/training%20notebooks/cons_vs_es.ipynb
# Don't blame me ¯\_(ツ)_/¯
def extract_values_from_dsc_np_format_and_write(array, path):
    lifehack = 5000000000
    class_task = True
    if class_task:
        # classification
        labels = array[:lifehack, 140, 0]
    else:
        # psi value
        labels = array[:lifehack, -1, 4]

    lens = array[:lifehack, -1, 0:3]
    # lens = lens[:, 0], lens[:, 1], lens[:, 2]
    start_seq, end_seq = array[:lifehack, :140, :4], array[:lifehack, 141:281, :4]
    start_seq, end_seq = decode_batch_of_seqs(start_seq), decode_batch_of_seqs(end_seq)

    to_return = []
    # could feed my network data with 280 + 3 + 1 dimensions
    with open(path, 'w') as f:
        for start, end, len, label in zip(start_seq, end_seq, lens, labels):
            f.write(f'{start}\t{end}\t{len[0]}\t{len[1]}\t{len[2]}\t{label}\n')


data_path = '../../data/distributed'
x_cons_data = np.load('../../data/hexevent/x_cons_data.npy')
hx_cas_data = np.load('../../data/hexevent/x_cas_data_high.npy')
lx_cas_data = np.load('../../data/hexevent/x_cas_data_low.npy')
x_cons_data[:,-1,4] = 1.

print('Loaded .npy data files')

extract_values_from_dsc_np_format_and_write(x_cons_data, f'{data_path}/decoded_cons_data_class.csv')
print('Decoded cons data')

extract_values_from_dsc_np_format_and_write(hx_cas_data, f'{data_path}/decoded_cas_data_high_class.csv')
print('Decoded high cass data')

extract_values_from_dsc_np_format_and_write(lx_cas_data, f'{data_path}/decoded_cas_data_low_class.csv')
print('Decoded low cass data')

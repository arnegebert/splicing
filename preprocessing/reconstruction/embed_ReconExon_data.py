import numpy as np
import csv
import gensim.models
import time
from utils import one_hot_decode_seq_vanilla

#todo could make this doable via command line arguments and export / document data processing process via a bash script
startt = time.time()
data_path = '../../data'

# src = 'brain_cortex_full.csv'
# target = 'embedded_gtex_junction_class.npy'
# avg_len = 7853.118899261425 # Junction length measurement
# std_len = 23917.691461462917


# src = 'gtex_processed/brain_cortex_cassette_full.csv'
# target = 'distributed/embedded_gtex_cass_class.npy'
# avg_len = 5028.584836672408 # Cassette exon measurements
# std_len = 18342.894894670942

src = 'dsc_reconstruction_junction/brain_cortex_full.csv'
target = 'dsc_reconstruction_exon/embedded_class.npy'
avg_len = 7770.843898366985 # Junction length first iteration reconstruction dataset
std_len = 21350.371079742523

embedding_model = gensim.models.Doc2Vec.load('../../model/d2v-full-5epochs')
classification_task = True
constitutive_level = 0.99

def batch_split_into_3_mers(batch):
    to_return = []
    for sentence in batch:
        words = []
        for i in range(1, len(sentence) - 1):
            words.append(sentence[i - 1:i + 2])
        words.append(to_return)
    return to_return

def split_into_3_mers(sentence):
    words = []
    for i in range(1, len(sentence) - 1):
        words.append(sentence[i - 1:i + 2])
    return words

print('Loading data')
cons_exons = np.load('../../data/dsc_reconstruction_exon/brain_cortex_cons.npy')
low_exons = np.load('../../data/dsc_reconstruction_exon/brain_cortex_low.npy')
high_exons = np.load('../../data/dsc_reconstruction_exon/brain_cortex_high.npy')
print('Finished loading data')

def decode_reshape_and_embed(batch):
    #batch = batch[:501]
    seq_len = len(batch[0])-1
    batch_vector = []
    # batch[:, 240, 3] = (batch[:, 240, 3] >= constitutive_level)#.astype(np.float32)
    # batch[:, 1200, 3] = (batch[:, 1200, 3] >= constitutive_level)#.astype(np.float32)
    batch[:, seq_len, 3] = (batch[:, seq_len, 3] >= constitutive_level)

    for i, line in enumerate(batch):
        if i % 500 == 0: print(f'Processing line {i}')
        # start_enc, end_enc = line[:140], line[140:280]
        # start_enc, end_enc = line[:600], line[600:1200]
        start_enc, end_enc = line[:seq_len//2], line[seq_len//2:seq_len]

        start_seq, end_seq = ''.join(one_hot_decode_seq_vanilla(start_enc)), \
                             ''.join(one_hot_decode_seq_vanilla(end_enc))
        start, end = split_into_3_mers(start_seq), split_into_3_mers(end_seq)
        start_d2v = embedding_model.infer_vector(start)
        end_d2v = embedding_model.infer_vector(end)
        dummy_vector = [0] * len(start_d2v)
        # dummy_vector[:4] = line[280]
        # dummy_vector[:4] = line[1200]
        dummy_vector[:4] = line[seq_len]
        data_vector = [start_d2v, end_d2v, dummy_vector]
        batch_vector.append(data_vector)

    batch_vector = np.array(batch_vector).astype(np.float32)
    return batch_vector

print('Beginning to decode, reshape and embed constitutive exon data')
cons_exons = decode_reshape_and_embed(cons_exons)
print('Cons data done')
low_exons = decode_reshape_and_embed(low_exons)
print('Low data done')
high_exons = decode_reshape_and_embed(high_exons)
print('High data done')

np.save(f'{data_path}/dsc_reconstruction_exon/embedded_brain_cortex_cons.npy', cons_exons)
np.save(f'{data_path}/dsc_reconstruction_exon/embedded_brain_cortex_low.npy', low_exons)
np.save(f'{data_path}/dsc_reconstruction_exon/embedded_brain_cortex_high.npy', high_exons)


endt = time.time()
print(f'Time to process data: {endt-startt}')
import time

import gensim.models
import numpy as np

from utils import one_hot_decode_seq_vanilla

startt = time.time()
data_path = '../../data'
data_dir = 'hexevent'
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
cons_exons = np.load(f'../../data/{data_dir}/cons.npy')
low_exons = np.load(f'../../data/{data_dir}/low2.npy')
high_exons = np.load(f'../../data/{data_dir}/high2.npy')
print('Finished loading data')

def decode_reshape_and_embed(batch):
    #batch = batch[:501]
    seq_len = len(batch[0])-1
    batch_vector = []
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

np.save(f'{data_path}/{data_dir}/embedded_cons.npy', cons_exons)
np.save(f'{data_path}/{data_dir}/embedded_low2.npy', low_exons)
np.save(f'{data_path}/{data_dir}/embedded_high2.npy', high_exons)


endt = time.time()
print(f'Time to process data: {endt-startt}')
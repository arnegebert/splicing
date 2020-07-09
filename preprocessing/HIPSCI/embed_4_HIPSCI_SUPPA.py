import numpy as np
import csv
import gensim.models
import time
from utils import one_hot_decode_seq_vanilla

#todo could make this doable via command line arguments and export / document data processing process via a bash script
startt = time.time()
data_path = '../../data'

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
cons_exons = np.load('../../data/hipsci_suppa/4_cons.npy')
low_exons = np.load('../../data/hipsci_suppa/4_low.npy')
high_exons = np.load('../../data/hipsci_suppa/4_high.npy')
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
        exon_before_enc, start_enc, end_enc, exon_after_enc = line[:seq_len//4], line[seq_len//4:2*seq_len//4], \
                                                       line[2*seq_len // 4:3 * seq_len // 4], line[3*seq_len//4:seq_len]

        start_seq, end_seq = ''.join(one_hot_decode_seq_vanilla(start_enc)), \
                             ''.join(one_hot_decode_seq_vanilla(end_enc))
        exon_before_seq, exon_after_seq = ''.join(one_hot_decode_seq_vanilla(exon_before_enc)),\
                                            ''.join(one_hot_decode_seq_vanilla(exon_after_enc))
        start, end = split_into_3_mers(start_seq), split_into_3_mers(end_seq)
        exon_before, exon_after = split_into_3_mers(exon_before_seq), split_into_3_mers(exon_after_seq)
        exon_before_d2v = embedding_model.infer_vector(exon_before)
        start_d2v = embedding_model.infer_vector(start)
        end_d2v = embedding_model.infer_vector(end)
        exon_after_d2v = embedding_model.infer_vector(exon_after)
        dummy_vector = [0] * len(start_d2v)
        # dummy_vector[:4] = line[280]
        # dummy_vector[:4] = line[1200]
        dummy_vector[:4] = line[seq_len]
        data_vector = [exon_before_d2v, start_d2v, end_d2v, exon_after_d2v, dummy_vector]
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

np.save(f'{data_path}/hipsci_suppa/4_embedded_cons.npy', cons_exons)
np.save(f'{data_path}/hipsci_suppa/4_embedded_low.npy', low_exons)
np.save(f'{data_path}/hipsci_suppa/4_embedded_high.npy', high_exons)


endt = time.time()
print(f'Time to process data: {endt-startt}')
import time

import gensim.models
import numpy as np

from utils import intron_mean, intron_std


"""

IMPORTANT:

TRIED REPLACING THIS BUT GOT REALLY WEIRD RESULTS WITH 0.5 AUC (network is always predicting the same thing - 0.4950)

THEREFORE KEPT THIS AND IT IS STILL USED

"""
#todo could make this doable via command line arguments and export / document data processing process via a bash script
startt = time.time()
# data_path = '../../data/gtex_processed'
data_path = '../../data'

# src = 'brain_cortex_full.csv'
# target = 'embedded_gtex_junction_class.npy'
# avg_len = 7853.118899261425 # Junction length measurement
# std_len = 23917.691461462917


# src = 'gtex_processed/brain_cortex_cassette_full.csv'
# target = 'distributed/embedded_gtex_cass_class.npy'
# avg_len = 5028.584836672408 # Cassette exon measurements
# std_len = 18342.894894670942
tissue = 'brain'
assert tissue in ['brain', 'cerebellum', 'heart']

if tissue == 'brain':
    src = 'dsc_reconstruction_junction/brain_cortex_full.csv'
    target = 'distributed/brain_cortex_embedded_recon_junc_class.npy'
elif tissue == 'heart':
    src = 'gtex_processed/heart_full.csv'
    target = 'distributed/heart_embedded_recon_junc_class.npy'
elif tissue == 'cerebellum':
    src = 'gtex_processed/cerebellum_full.csv'
    target = 'distributed/cerebellum_embedded_recon_junc_class.npy'
# avg_len = 7770.843898366985 # Junction length first iteration reconstruction dataset
# std_len = 21350.371079742523

embedding_model = gensim.models.Doc2Vec.load('../../model/d2v-full-5epochs')
classification_task = True
constitutive_level = 0.95

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

low = 0
medium = 0
high = 0
cons = 0
samples = []

with open(f'{data_path}/{src}') as f:
    for i, l in enumerate(f):
        if i % 1000==0: print(f'Processing line {i}')
        j, start_seq, end_seq, psi = l.split(',')
        s, e = j.split('_')[1:3]
        s, e = int(s), int(e)
        psi = float(psi[:-1])
        is_constitutive = float(psi >= constitutive_level)

        if psi <= 0.2: low += 1
        if 0.2 < psi <= 0.75: medium += 1
        if psi > 0.75 and psi < constitutive_level: high += 1
        if psi > constitutive_level: cons += 1

        label = is_constitutive if classification_task else psi

        l1 = (e - s - intron_mean) / intron_std
        start, end = split_into_3_mers(start_seq), split_into_3_mers(end_seq)
        start_d2v = embedding_model.infer_vector(start)
        end_d2v = embedding_model.infer_vector(end)
        dummy_vector = [0] * len(start_d2v)
        dummy_vector[0] = l1
        dummy_vector[3] = label

        data_vector = [start_d2v, end_d2v, dummy_vector]
        samples.append(data_vector)
        # if len(samples) == 3: break

print(f'low: {low}')
print(f'medium: {medium}')
print(f'high: {high}')
print(f'cons: {cons}')
print(f'all: {low+medium+high+cons}')

total = low+medium+high+cons
cons_perc = cons / total
print(f'Percentage of consecutive data: {cons_perc}')
if cons_perc > 0.6 or cons_perc < 0.4:
    raise Exception('Unbalanced dataset')

samples = np.array(samples).astype(np.float32)
np.save(f'../../data/{target}', samples)

def embed_and_reshape(decoded_data):
    # batch_vector = np.array([])
    batch_vector = []
    for start, end, l1, l2, l3, label in decoded_data:
        start, end = split_into_3_mers(start), split_into_3_mers(end)
        start_d2v = embedding_model.infer_vector(start)
        end_d2v = embedding_model.infer_vector(end)
        dummy_vector = [0] * len(start_d2v)
        # dummy_vector = np.zeros_like(start_d2v)
        dummy_vector[0] = l1
        dummy_vector[1] = l2
        dummy_vector[2] = l3
        dummy_vector[3] = label

        data_vector = [start_d2v, end_d2v, dummy_vector]
        batch_vector.append(data_vector)
        # data_vector = np.concatenate((start_d2v, end_d2v, dummy_vector))
        # batch_vector = np.append(batch_vector, data_vector)
    batch_vector = np.array(batch_vector).astype(np.float32)
    #batch_vector = np.reshape(batch_vector, (-1, 100, 3))
    return batch_vector

# x_cons_data = embed_and_reshape(x_cons_data)
# print('Cons data done')
# hx_cas_data = embed_and_reshape(hx_cas_data)
# print('High data done')
# lx_cas_data = embed_and_reshape(lx_cas_data)
# print('Low data done')
#
# np.save(f'{data_path}/embedded_cons_data_class.npy', x_cons_data)
# np.save(f'{data_path}/embedded_cas_data_high_class.npy', hx_cas_data)
# np.save(f'{data_path}/embedded_cas_data_low_class.npy', lx_cas_data)


endt = time.time()
print(f'Time to process data: {endt-startt}')
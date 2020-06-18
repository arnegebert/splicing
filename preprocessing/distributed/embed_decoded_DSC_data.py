import numpy as np
import csv
import gensim.models

data_path = '../../data/distributed'
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

with open(f'{data_path}/decoded_cons_data_class.csv') as f:
    reader = csv.reader(f, delimiter='\t')
    x_cons_data = list(reader)
x_cons_data = np.array(x_cons_data)
with open(f'{data_path}/decoded_cas_data_high_class.csv') as f:
    reader = csv.reader(f, delimiter='\t')
    hx_cas_data = list(reader)
hx_cas_data = np.array(hx_cas_data)
with open(f'{data_path}/decoded_cas_data_low_class.csv') as f:
    reader = csv.reader(f, delimiter='\t')
    lx_cas_data = list(reader)
lx_cas_data = np.array(lx_cas_data)


embedding_model = gensim.models.Doc2Vec.load('../../model/d2v-full-5epochs')

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
    batch_vector = np.array(batch_vector)
    batch_vector = np.reshape(batch_vector, (-1, 100, 3))
    return batch_vector

x_cons_data = embed_and_reshape(x_cons_data)
hx_cas_data = embed_and_reshape(hx_cas_data)
lx_cas_data = embed_and_reshape(lx_cas_data)

np.save(f'{data_path}/embedded_cons_data_class.npy', x_cons_data)
np.save(f'{data_path}/embedded_cas_data_high_class.npy', hx_cas_data)
np.save(f'{data_path}/embedded_cas_data_low_class.npy', lx_cas_data)

# with open(f'{data_path}/embedded_cas_data_high_class.npy', 'w') as f:
#     for start, end, l1, l2, l3, label in hx_cas_data:
#         start, end = split_into_3_mers(start), split_into_3_mers(end)
#         start_d2v = embedding_model.infer_vector(start)
#         end_d2v = embedding_model.infer_vector(end)
#         dummy_vector = np.zeros_like(start_d2v)
#         dummy_vector[0] = l1
#         dummy_vector[1] = l2
#         dummy_vector[2] = l3
#         dummy_vector[3] = label
#         data_vector = np.concatenate(start_d2v, end_d2v, dummy_vector)
#         np.save(f'{data_path}/embedded_cas_data_high_class.npy', data_vector)
#         f.write(f'{start}\t{end}\t{l1}\t{l2}\t{l3}\t{label}\n')


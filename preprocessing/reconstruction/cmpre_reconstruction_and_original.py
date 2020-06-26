import numpy as np
from utils import one_hot_encode_seq, one_hot_decode_seq_vanilla

data_path = '../../data'
cons_recons_path = 'dsc_reconstruction_exon/brain_cortex_cons.npy'

cons_recons = np.load(f'{data_path}/{cons_recons_path}')
cons_original = np.load(f'{data_path}/hexevent/x_cons_data.npy')
print('Data loaded')

recon_lens, recon_target = cons_recons[0, 280, :3], cons_recons[0, 280, 3]

orig_lens, orig_target = cons_original[:, -1, :3], cons_original[:, -1, 3]

eps = 1e-3
def kms(lens_gai, lens_fake):
    l1, l2, l3 = lens_gai
    l1, l2, l3 = float(l1), float(l2), float(l3)
    l4, l5, l6 = lens_fake
    l4, l5, l6 = float(l4), float(l5), float(l6)
    b1 = abs(l1 - l4 ) < eps
    b2 = abs(l2 - l5 ) < eps
    b3 = abs(l3 - l6 ) < eps
    return b1 and b2 and b3
    #return (l1, l2, l3) == (l4, l5, l6)

cnter = 0
for i, lens in enumerate(orig_lens):
    if i % 1000 == 0: print(f'Line {i}')
    if kms(recon_lens, lens):
    # if (lens == recon_lens).all():
        cnter += 1
        # print('xxx')
        start_seq_orig, end_seq_orig = cons_original[i, :140, :4], cons_original[i, 141:281, :4]
        start_seq_recon, end_seq_recon = cons_recons[0, :140, :4], cons_recons[0, 140:280, :4]

        x1 = sum(start_seq_orig == start_seq_recon)
        x2 = sum(end_seq_orig == end_seq_recon)
        # print(x1, x2)

print(f'Number of length matches: {cnter}')

# hashs = set()
# dupl = 0
# orig_start_seqs = cons_original[:, :140, :4]
# for s in orig_start_seqs:
#     s = tuple(map(tuple, s))
#     if s in hashs:
#         dupl += 1
#     else:
#         hashs.add(s)
# print(dupl)

seq = 'AAGCTTA'
enc = np.array(one_hot_encode_seq(seq))
dec = ''.join(one_hot_decode_seq_vanilla(enc))

print(seq)
print(dec)
print(seq == dec)
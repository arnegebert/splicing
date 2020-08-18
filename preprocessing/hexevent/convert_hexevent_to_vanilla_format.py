import numpy as np

src = '../../data/hexevent'
trgt = '../../data/hexevent'

cons = np.load(f'{src}/cons_original_format.npy')
high = np.load(f'{src}/high_original_format.npy')
low = np.load(f'{src}/low_original_format.npy')

# original data somehow has 0 there
cons[:, -1, 4] = 1

def convert_hexevent_to_vanilla_format(array):
    lifehack = 500000
    psi = array[:lifehack, -1, 4]
    start_seq, end_seq = array[:lifehack, :140, :4], array[:lifehack, 141:281, :4]
    start_and_end = np.concatenate((start_seq, end_seq), axis=1)
    lens = array[:lifehack, -1, 0:3]
    lens_and_psi = np.concatenate((lens, psi.reshape(-1, 1)), axis=1).reshape(-1, 1, 4)
    samples = np.concatenate((start_and_end, lens_and_psi), axis=1).astype(np.float32)
    return samples

cons = convert_hexevent_to_vanilla_format(cons)
low = convert_hexevent_to_vanilla_format(low)
high = convert_hexevent_to_vanilla_format(high)

total = len(cons) + len(high) + len(low)
print(f'Total number of exons: {total}')
print(f'Number of low PSI exons: {len(cons)}')
print(f'Number of high PSI exons: {len(high)}')
print(f'Number of cons exons: {len(low)}')

np.save(f'{src}/cons.npy', cons)
np.save(f'{src}/low.npy', low)
np.save(f'{src}/high.npy', high)
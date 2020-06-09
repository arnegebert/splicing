import numpy as np
from torch import as_tensor as T, Tensor

# fifth column (I assume the inclusion level?) for this is often 0
cons = np.load('../../data/hexevent/x_cons_data.npy')
# fifth column (I assume the inclusion level) for this is often close to 1
cass_low = np.load('../../data/hexevent/x_cas_data_low.npy')


def extract_values_from_dsc_np_format(array):
    psi = array[:500, 140, 0]
    start_seq, end_seq = array[:500, :140, :4], array[:, 141:281, :4]
    lens = array[:500, -1, 0:3]
    to_return = []
    for s, e, l, p in zip(start_seq, end_seq, lens, psi):
        to_return.append((T((s, e)).float(), T(l).float(), T(p).float()))
    return to_return

x = extract_values_from_dsc_np_format(cons)
print('xxx')
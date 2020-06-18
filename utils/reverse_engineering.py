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

def decode_seq_vanilla(seq):
    to_return = []
    for encoding in seq:
        to_return.append(one_hot_decode(encoding))
    return to_return

def decode_seq(seq, dct):
    to_return = []
    for encoding in seq:
        to_return.append(dct[encoding])
    return to_return


x_cons_data = np.load('data/hexevent/x_cons_data.npy')
tester = x_cons_data[0, :140, :4]
decoded = decode_seq_vanilla(tester)
decoded = ''.join(decoded)
print('plis')

legit_sequences = load_start_of_legit_sequences()

perms = generate_all_permutations('ACGT')

for i, p in enumerate(perms):
    print(f'Testing permutation {i}: {p}')
    dct = build_dictionary_off_permutation(p)
    decoded = decode_seq(tester, dct)
    decoded = ''.join(decoded)
    for seq in legit_sequences:
        extract = decoded[5:55]
        if extract in seq:
            print('EUREKA')

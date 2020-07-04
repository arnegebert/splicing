def reverse_complement(seq):
    complt = []
    for bp in seq:
        if bp == 'a' or bp == 'A':
            complt.append('T')
        elif bp == 'c' or bp == 'C':
            complt.append('G')
        elif bp == 'g' or bp == 'G':
            complt.append('C')
        elif bp == 't' or bp == 'T':
            complt.append('A')
        else: raise Exception("Unidentified base-pair given")
    return ''.join(complt)

def one_hot_encode(nt):
    if nt == 'A' or nt == 'a':
        return [1.0, 0, 0, 0]
    elif nt == 'C' or nt == 'c':
        return [0, 1.0, 0, 0]
    elif nt == 'G' or nt == 'g':
        return [0, 0, 1.0, 0]
    elif nt == 'T' or nt == 't':
        return [0, 0, 0, 1.0]
    else: raise Exception('Unknown nucleotide given')

def one_hot_encode_seq(seq):
    encoding = []
    for nt in seq:
        encoding.append(one_hot_encode(nt))
    return encoding

def one_hot_decode(nt):
    if (nt == [1.0, 0, 0, 0]).all():
        return 'A'
    elif (nt == [0, 1.0, 0, 0]).all():
        return 'C'
    elif (nt == [0, 0, 1.0, 0]).all():
        return 'G'
    elif (nt == [0, 0, 0, 1.0]).all():
        return 'T'

def one_hot_decode_seq_vanilla(seq):
    to_return = []
    for encoding in seq:
        to_return.append(one_hot_decode(encoding))
    return to_return


def reverse_complement(seq):
    complt = []
    for bp in seq:
        if bp == 'a' or bp == 'A':
            complt.append('t')
        elif bp == 'c' or bp == 'C':
            complt.append('g')
        elif bp == 'g' or bp == 'G':
            complt.append('c')
        elif bp == 't' or bp == 'T':
            complt.append('a')
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

def one_hot_encode_seq(seq):
    encoding = []
    for nt in seq:
        encoding.append(one_hot_encode(nt))
    return encoding
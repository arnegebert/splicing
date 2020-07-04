import numpy as np

psis = []
exons = []

with open('second.psi') as f:
    for i, l in enumerate(f):
        if i==0: continue
        if i % 1000 == 0: # ~ 357500 junctions
            print(f'Reading line {i}')

        eventid, psi = l.split('\t')
        psi = psi[:-1]
        if psi in ['nan', 'NA']: continue
        try:
            psi =float(psi)
        except ValueError:
            continue
        geneid_eventtype, chrom, coord1, coord2, strand = eventid.split(':')
        try:
            read_chrom = int(chrom[3:])
        except ValueError:
            continue
        e1, s1 = coord1.split('-')
        e2, s2 = coord1.split('-')
        exons.append((int(chrom[3:]), strand, int(e1), int(s1), int(e2), int(s2), psi))
        psis.append(psi)

exons.sort(key=lambda tup: tup[5])
exons.sort(key=lambda tup: tup[4])
exons.sort(key=lambda tup: tup[3])
exons.sort(key=lambda tup: tup[2])
exons.sort(key=lambda tup: tup[0])

psis = np.array(psis)
print(len(psis))
print(np.mean(psis))
print(np.median(psis))

with open('formatted_second.psi', 'w') as f:
    for (chrom, strand, e1, s1, e2, s2, psi) in exons:
        f.write(f'{chrom}\t{strand}\t{e1}\t{s1}\t{e2}\t{s2}\t{psi}\n')
        #f.write('\t'.join(exon))
# with vs without formatting doesn't make a difference
# 18360
# 0.4743559224421955
# 0.2023384176178008
# sum(events==1) = 7900
# sum(events==0) = 8792
# sum(events>=0.99) = 7955
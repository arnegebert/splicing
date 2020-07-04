import numpy as np

psis = []
with open('second.psi') as f:
    for i, l in enumerate(f):
        if i==0: continue
        eventid, psi = l.split('\t')
        psi = psi[:-1]
        if psi in ['nan', 'NA']: continue
        try:
            psi =float(psi)
        except ValueError:
            continue
        psis.append(psi)

psis = np.array(psis)
print(len(psis))
print(np.mean(psis))
print(np.median(psis))

# with vs without formatting doesn't make a difference
# 18360
# 0.4743559224421955
# 0.2023384176178008
# sum(events==1) = 7900
# sum(events==0) = 8792
# sum(events>=0.99) = 7955
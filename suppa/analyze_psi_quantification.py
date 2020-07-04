import numpy as np

events = []
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
        events.append(psi)

events = np.array(events)
print(len(events))
print(np.mean(events))
print(np.median(events))

# with vs without formatting doesn't make a difference
# 18360
# 0.4743559224421955
# 0.2023384176178008
# sum(events==1) = 7900
# sum(events==0) = 8792
# sum(events>=0.99) = 7955
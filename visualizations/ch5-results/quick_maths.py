import numpy as np

x = np.array([0.9940236726287854, 0.9805555722786428, 0.9873636495598455, 0.9910209103056676, 0.9950046555830635,
              0.9808057621012873, 0.9890807004029947, 0.9961606708244893, 0.982223911869075])

print(np.mean(x))




x = np.array([
0.9553371837649459, 0.9904798558552291, 0.9550468090083868, 0.9729119626057124,
0.9873637183134901, 0.9597310119907871, 0.9922904605341837, 0.9654297499738087,
0.9784849844502715])
print(np.mean(x))


bnn_udc = np.array([21, 22, 24, 21, 22])
bnn_lmh = np.array([32, 37, 37, 38, 40])
dnn_psi = np.array([41, 47, 40, 40, 49])
d2v = np.array([57, 72, 61, 66, 41])
w2v = np.array([64, 75, 69, 79, 53])

print(f'BNN-UDC & {np.mean(bnn_udc)}')
print(f'BNN-LMH & {np.mean(bnn_lmh)}')
print(f'DNN & {np.mean(dnn_psi)}')
print(f'D2V & {np.mean(d2v)}')
print(f'W2V & {np.mean(w2v)}')

def compute_relative_performance_change(performance_bef, performance_after):
    return (performance_after - performance_bef)/(performance_after-0.5)

dsc_bef, dsc_after = 0.661, 0.704
d2v_bef, d2v_after = 0.629, 0.673
rasc_bef, rasc_after = 0.776, 0.808

dsc_change = compute_relative_performance_change(dsc_bef, dsc_after)
d2v_change = compute_relative_performance_change(d2v_bef, d2v_after)
rasc_change = compute_relative_performance_change(rasc_bef, rasc_after)

print(f'DSC Performance change: {dsc_change:.3f}')
print(f'D2V Performance change: {d2v_change:.3f}')
print(f'RASC Performance change: {rasc_change:.3f}')

dsc_perf, rasc_perf = 0.822, 0.875
rasc_improv = compute_relative_performance_change(dsc_perf, rasc_perf)
print(f'RASC improves upon DSC by {rasc_improv:.3f}')


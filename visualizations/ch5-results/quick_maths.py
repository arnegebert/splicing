import numpy as np

x = np.array([0.9940236726287854, 0.9805555722786428, 0.9873636495598455, 0.9910209103056676, 0.9950046555830635,
              0.9808057621012873, 0.9890807004029947, 0.9961606708244893, 0.982223911869075])

print(np.mean(x))




test_ipsc_d2v = np.array([0.7019190641985806, 0.7019190641985806, 0.7013794001488541, 0.7233028134100813, 0.7152093500072498, 0.7113968310742923, 0.7064846101443125, 0.7179934819897874, 0.7091494155785326, 0.7109347294928198])
diff_lib_ipsc_d2v = np.array([ 0.6691303220745322, 0.6691303220745322, 0.6708333160810134, 0.6754636401155479, 0.6751865134063197, 0.6703340110122954, 0.6736248325922735, 0.6716030232093311, 0.6689852018791418, 0.6711797747335007])
diff_indv_ipsc_d2v = np.array([0.6710851979518251, 0.6710851979518251, 0.6732087991361819, 0.6777442885560986, 0.6778857819866774, 0.6735407848120347, 0.6771481847433974, 0.6747756110870683, 0.6717970042934985, 0.6741467865488221])
diff_tiss_ipsc_d2v = np.array([0.7084954114659393, 0.7084954114659393, 0.7155054813930671, 0.7167098195997647, 0.712093785301092, 0.7134214121960812, 0.717629406495572, 0.7162154779776815, 0.7147319314083559, 0.7149097389855633])

def print_mean_and_std(values):
    print(f'{np.mean(values):.3f} +- {np.std(values):.3f}')

print_mean_and_std(test_ipsc_d2v)
print_mean_and_std(diff_lib_ipsc_d2v)
print_mean_and_std(diff_indv_ipsc_d2v)
print_mean_and_std(diff_tiss_ipsc_d2v)


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

# cons: 0.590%, low: 0.183%, high: 0.227%
# cons: 0.564%, low: 0.192%, high: 0.244% or 44 vs 56
cons, low, high = 26394,  8201 , 10151
# cons, low, high = 27371, 9299, 11819


total = cons + low + high
print(f'cons: {cons / total:.3f}%')
print(f'low: {low / total:.3f}%')
print(f'high: {high / total:.3f}%')

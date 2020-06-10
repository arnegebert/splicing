import numpy as np
import matplotlib.pyplot as plt

def is_outlier(points, thresh=10):
    """
    Returns a boolean array with True if points are outliers and False
    otherwise.

    Parameters:
    -----------
        points : An numobservations by numdimensions array of observations
        thresh : The modified z-score to use as a threshold. Observations with
            a modified z-score (based on the median absolute deviation) greater
            than this value will be classified as outliers.

    Returns:
    --------
        mask : A numobservations-length boolean array.

    References:
    ----------
        Boris Iglewicz and David Hoaglin (1993), "Volume 16: How to Detect and
        Handle Outliers", The ASQC Basic References in Quality Control:
        Statistical Techniques, Edward F. Mykytka, Ph.D., Editor.
    """
    if len(points.shape) == 1:
        points = points[:,None]
    median = np.median(points, axis=0)
    diff = np.sum((points - median)**2, axis=-1)
    diff = np.sqrt(diff)
    med_abs_deviation = np.median(diff)

    modified_z_score = 0.6745 * diff / med_abs_deviation

    return modified_z_score > thresh

tpms = []
with open('../data/brain_cortex_tpm_one_sample.csv') as f:
    for l in f:
        # j, start_seq, end_seq, psi = l.split(',')
        gene, tpm = l.split(',')
        tpms.append(float(tpm[:-1]))

tpms = np.array(tpms)
length = len(tpms)
print(f'number of values: {length}')
print(f'percent of zeroes: {np.sum(tpms == 0) / length}')
# print(f'percent of psis <0.5: {np.sum(psis<0.5)/length}')
# print(f'percent of psis =0.5: {np.sum(psis==0.5)/length}')
# print(f'percent of psis >0.5: {np.sum(psis>0.5)/length}')
# print(f'percent of psis =1: {np.sum(psis==1)/length}')
print(f'mean: {np.mean(tpms)}')
print(f'median: {np.median(tpms)}')

filtered = tpms[tpms>=10]
filtered = filtered[filtered<500]
# filtered = filtered[~is_outlier(filtered)]

plt.hist(filtered)
plt.xlabel('TPM value')
plt.ylabel('number of data points')
plt.show()

print(f'median after filtering: {np.median(filtered)}')

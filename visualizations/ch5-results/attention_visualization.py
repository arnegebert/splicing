import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from utils import one_hot_decode_seq_vanilla

sample = 2
plt.style.use('seaborn')

data = np.load(f'attn_ws/val_all_data.npy')[:, :, 0]
attn_ws = np.load(f'attn_ws/attn_ws_50.npy')[:, :, 0]
data = one_hot_decode_seq_vanilla(data[sample][:-1])
mean, std = np.mean(attn_ws, axis=0), np.std(attn_ws, axis=0)
stderr = stats.sem(attn_ws, axis=0)
# if False:
#     attn_ws = attn_ws[sample, :]
# else:
#     attn_ws = np.mean(attn_ws, axis=0)

print(f'Mean std dev: {np.mean(np.std(attn_ws, axis=0))}')
# fig, axs = plt.subplots(1, 2, sharey=True)
# axs[0].plot(attn_ws[:140])
# axs[1].plot(attn_ws[140:])
fig, ax = plt.subplots()

if True:
    #x_ticks = [i for i in range(-140, 141) if i]#np.arange(-140, -1, -1)
    ax.set_xlim(0, 280)
    # plt.gca().set_ylim(ymin=0)
    xticks = np.linspace(0, 280, 5)
    xs= ['start-70', 'exon start', 'exon start/end transition', 'exon end', 'end+70']
    plt.xticks(xticks, xs)
    plt.rcParams['xtick.major.size'] = 20
    plt.rcParams['xtick.major.width'] = 4
    plt.rcParams['xtick.bottom'] = True
    plt.rcParams['ytick.left'] = True

    ax.errorbar(x=range(0, 280), y=mean, yerr = stderr)
    # ax.errorbar(x=range(0, 280), y=mean)
    ax.set_ylim(bottom=0, top=None)


    # ax.annotate('Exon start',
    #             xy=(0, 0),
    #             xytext=(0, 0.001),
    #             arrowprops = dict(facecolor='black', shrink=0.05))
else:
    plt.xticks(range(280), data)
    ax.plot(attn_ws, )


plt.savefig('mean_attention_w_std.png',bbox='tight')

plt.show()
print(attn_ws.shape)
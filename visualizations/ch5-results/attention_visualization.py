import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from utils import one_hot_decode_seq_vanilla

def line_plot(attn_ws):
    plt.style.use('seaborn')
    mean, std = np.mean(attn_ws, axis=0), np.std(attn_ws, axis=0)

    # print(f'Mean std dev: {np.mean(np.std(attn_ws, axis=0))}')
    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
    # xticks = np.linspace(0, 140, 5)
    # xs= ['-70', '-35', 'start', '+35', '+70']
    xs2 = np.arange(-70, 70)
    ax1.set_xlim(-70, 70)
    ax1.set_xlabel('Position relative to exon start')
    ax2.set_xlabel('Position relative to exon end')
    ax1.set_ylabel('Attention weight')

    ax2.set_xlim(-70, 70)
    # xticks2 = np.linspace(-70, 70, 5)
    # ax1.set_xticks(xticks2, xs)
    # plt.xticks(xticks2, xs)
    ax1.errorbar(xs2, mean[:140])#, yerr=stderr[:140])
    ax2.errorbar(xs2, mean[140:])#, yerr=stderr[140:])

    plt.savefig('mean_attention_w_std.png', dpi=300, bbox='tight')

    plt.show(dpi=300)
    # print(attn_ws.shape)


def heatmap():
    # plt.style.use('seaborn')


    mean_attn_ws_epochs = []
    for i in range(1, 50):
        attn_ws = np.load(f'attn_ws/attn_ws_{i}.npy')[:, :140, 0]

        mean, std = np.mean(attn_ws, axis=0), np.std(attn_ws, axis=0)
        mean_attn_ws_epochs.append(mean)
    mean_attn_ws_epochs = np.concatenate(mean_attn_ws_epochs, axis=0).reshape(49, 140)

    plt.imshow(mean_attn_ws_epochs, cmap='viridis')
    plt.savefig('attention_heatmap.png', dpi=300, bbox='tight')

    plt.show(dpi=300)

def print_attn_sums(attn_ws):
    print(f'{sum(attn_ws[0, :140])} vs {sum(attn_ws[0, 140:])}')

heatmap()

attn_ws = np.load(f'attn_ws/attn_ws_50.npy')[:, :, 0]
print_attn_sums(attn_ws)
line_plot(attn_ws)



attn_ws_multi_heads = np.load(f'../../attn_ws_50.npy')
attn_ws_0 = attn_ws_multi_heads[:, :, 0]
print_attn_sums(attn_ws_0)
line_plot(attn_ws_0)
attn_ws_1 = attn_ws_multi_heads[:, :, 1]
print_attn_sums(attn_ws_1)
line_plot(attn_ws_1)
attn_ws_2 = attn_ws_multi_heads[:, :, 2]
print_attn_sums(attn_ws_2)
line_plot(attn_ws_2)
attn_ws_3 = attn_ws_multi_heads[:, :, 3]
print_attn_sums(attn_ws_3)
line_plot(attn_ws_3)
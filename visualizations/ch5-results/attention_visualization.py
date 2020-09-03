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


def bar_chart(attn_ws):
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
    ax1.bar(xs2, mean[:140])#, yerr=stderr[:140])
    ax2.bar(xs2, mean[140:])#, yerr=stderr[140:])

    plt.savefig('mean_attention_barchart.png', dpi=300, bbox='tight')

    plt.show(dpi=300)
    # print(attn_ws.shape)

def heatmap2():
    # plt.style.use('seaborn')


    mean_attn_ws_epochs = []
    epochs = 109
    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
    for i in range(epochs):
        attn_ws = np.load(f'attn_ws_multi_distributed/attn_ws_{i+1}.npy')#[:,:140]
        # averaging across the attention heads
        attn_ws = np.mean(attn_ws, axis=-1)

        mean = np.mean(attn_ws, axis=0)
        mean_attn_ws_epochs.append(mean)
    mean_attn_ws_epochs = np.concatenate(mean_attn_ws_epochs, axis=0).reshape(epochs, 280)

    ax1.imshow(mean_attn_ws_epochs[:, :140], cmap='viridis')
    ax2.imshow(mean_attn_ws_epochs[:, 140:], cmap='viridis')
    plt.savefig('attention_heatmap.png', dpi=300, bbox='tight')
    plt.tight_layout()
    plt.show(dpi=300)

def heatmap():
    # plt.style.use('seaborn')

    attn_ws_epochs = []
    epochs = 109
    for i in range(epochs):
        attn_ws = np.load(f'attn_ws_multi_distributed/attn_ws_{i+1}.npy')#[:,:140]
        # averaging across the attention heads
        attn_ws = np.mean(attn_ws, axis=-1)

        mean = np.mean(attn_ws, axis=0)
        # attn_ws_epochs.append(mean)
        attn_ws_epochs.append(attn_ws[7])
    attn_ws_epochs = np.concatenate(attn_ws_epochs, axis=0).reshape(epochs, 280)

    plt.imshow(attn_ws_epochs[:, :140], cmap='viridis', interpolation='nearest')
    # plt.xlim(0, 139)
    plt.savefig('attention_heatmap_start.png', dpi=300, bbox='tight')
    plt.show(dpi=300)

    plt.imshow(attn_ws_epochs[:, 140:], cmap='viridis', interpolation='nearest')
    # plt.xlim(140)
    plt.savefig('attention_heatmap_end.png', dpi=300, bbox='tight')
    plt.show(dpi=300)

def print_attn_sums(attn_ws):
    print(f'{sum(np.mean(attn_ws[:, :140], axis=0)):.2f}')
          # f' vs'
          # f' {sum(attn_ws[:, 140:]):.2f}')

# heatmap()

# attn_ws = np.load(f'attn_ws/attn_ws_50.npy')[:, :, 0]
# print_attn_sums(attn_ws)
# line_plot(attn_ws)

# attn_ws = np.load(f'attn_ws/attn_ws_50.npy')[:, :, 0]
# print_attn_sums(attn_ws)
# bar_chart(attn_ws)

attn_ws_multi_heads = np.load(f'attn_ws_multi_distributed/attn_ws_110.npy')
# attn_ws_0 = attn_ws_multi_heads[:, :, 0]
# print_attn_sums(attn_ws_0)
# line_plot(attn_ws_0)
# attn_ws_1 = attn_ws_multi_heads[:, :, 1]
# print_attn_sums(attn_ws_1)
# line_plot(attn_ws_1)
# attn_ws_2 = attn_ws_multi_heads[:, :, 2]
# print_attn_sums(attn_ws_2)
# line_plot(attn_ws_2)
# attn_ws_3 = attn_ws_multi_heads[:, :, 3]
# print_attn_sums(attn_ws_3)
# line_plot(attn_ws_3)
#
# mean_attn_ws_multi_heads = np.mean(attn_ws_multi_heads, axis=-1)
# line_plot(mean_attn_ws_multi_heads)

mean_attn_ws_multi_heads = np.mean(attn_ws_multi_heads, axis=-1)
bar_chart(mean_attn_ws_multi_heads)

# for i in range(1, 100):
#     attn_ws_multi_heads = np.load(f'../../attn_ws_{i}.npy')
#     attn_ws_0 = attn_ws_multi_heads[:, :, 0]
#     print(f'{i}:{sum(np.mean(attn_ws_0[:, :140], axis=0)):.2f}')
#     # print_attn_sums(attn_ws_multi_heads[:, :, 0])

"""
Problem: doesn't really come out nicely if I normalize attention over whole 280 inputs / try to have 2 separate graphs
---> decision to not look at average but rather the 
"""
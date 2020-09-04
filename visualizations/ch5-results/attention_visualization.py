import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from utils import one_hot_decode_seq_vanilla
from mpl_toolkits.axes_grid1 import ImageGrid

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

def heatmap3():
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

def heatmap2(target=None):
    # observation: no difference between constitutive and non-constitutive exons attention
    # plt.style.use('seaborn')

    attn_ws_epochs = []
    epochs = 109

    if target is not None:
        test_set = np.load('attn_ws_multi_distributed/test_all_data.npy')

    for i in range(epochs):
        attn_ws = np.load(f'attn_ws_multi_distributed/attn_ws_{i+1}.npy')#[:,:140]
        # averaging across the attention heads
        attn_ws = np.mean(attn_ws, axis=-1)
        if target is not None:
            attn_ws = attn_ws[test_set[:,280,3]==target]
        # mean = np.mean(attn_ws, axis=0)
        # attn_ws_epochs.append(mean)
        attn_ws_epochs.append(attn_ws[7])
    attn_ws_epochs = np.concatenate(attn_ws_epochs, axis=0).reshape(epochs, 280)

    plt.imshow(attn_ws_epochs[:, :140], cmap='viridis', interpolation='nearest')
    # plt.xlim(0, 139)
    plt.title('First 140 nucleotides')
    plt.savefig('attention_heatmap_start.png', dpi=300, bbox='tight')
    plt.show(dpi=300)

    plt.imshow(attn_ws_epochs[:, 140:], cmap='viridis', interpolation='nearest')
    # plt.xlim(140)
    plt.title('Last 140 nucleotides')
    plt.savefig('attention_heatmap_end.png', dpi=300, bbox='tight')
    plt.show(dpi=300)


def heatmap4():
    # plt.style.use('seaborn')


    mean_attn_ws_epochs = []
    epochs = 109
    fig, ax = plt.subplots(2, 4, sharey=True, sharex=True)
    for i in range(epochs):
        attn_ws = np.load(f'attn_ws_multi_distributed/attn_ws_{i+1}.npy')#[:,:140]

        mean = np.mean(attn_ws, axis=0)
        mean_attn_ws_epochs.append(mean)
    mean_attn_ws_epochs = np.concatenate(mean_attn_ws_epochs, axis=0).reshape(epochs, 280, 4)

    ax[0, 0].imshow(mean_attn_ws_epochs[:, :140, 0], cmap='viridis')
    ax[0, 1].imshow(mean_attn_ws_epochs[:, 140:, 0], cmap='viridis')
    ax[0, 2].imshow(mean_attn_ws_epochs[:, :140, 1], cmap='viridis')
    ax[0, 3].imshow(mean_attn_ws_epochs[:, 140:, 1], cmap='viridis')
    ax[1, 0].imshow(mean_attn_ws_epochs[:, :140, 2], cmap='viridis')
    ax[1, 1].imshow(mean_attn_ws_epochs[:, 140:, 2], cmap='viridis')
    ax[1, 2].imshow(mean_attn_ws_epochs[:, :140, 2], cmap='viridis')
    ax[1, 3].imshow(mean_attn_ws_epochs[:, 140:, 2], cmap='viridis')
    plt.savefig('attention_heatmap.png', dpi=300, bbox='tight')
    plt.tight_layout()
    plt.show(dpi=300)


def heatmap():
    # plt.style.use('seaborn')

    mean_attn_ws_epochs = []
    epochs = 109
    for i in range(epochs):
        attn_ws = np.load(f'attn_ws_multi_distributed/attn_ws_{i+1}.npy')#[:,:140]

        mean = np.mean(attn_ws, axis=0)
        mean_attn_ws_epochs.append(mean)
    mean_attn_ws_epochs = np.concatenate(mean_attn_ws_epochs, axis=0).reshape(epochs, 280, 4)

    im1, im2 = mean_attn_ws_epochs[:, :140, 0], mean_attn_ws_epochs[:, 140:, 0]
    im3, im4 = mean_attn_ws_epochs[:, :140, 1], mean_attn_ws_epochs[:, 140:, 1]
    im5, im6 = mean_attn_ws_epochs[:, :140, 2], mean_attn_ws_epochs[:, 140:, 2]
    im7, im8 = mean_attn_ws_epochs[:, :140, 3], mean_attn_ws_epochs[:, 140:, 3]
    fig = plt.figure(figsize=(16., 8.))
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                     nrows_ncols=(2, 4),  # creates 2x2 grid of axes
                     axes_pad=0.2,  # pad between axes in inch.
                     )

    for i, (ax, im) in enumerate(zip(grid, [im1, im2, im3, im4, im5, im6, im7, im8])):
        # Iterating over the grid returns the Axes.
        ax.imshow(im, extent=[-70, 70, epochs, 0])


    # grid[0].set_xticks([], [])

    grid[0].set_ylabel('Epochs')
    grid[4].set_ylabel('Epochs')

    xs = np.arange(-70, 70)
    xticks = np.linspace(-70, 70, 5)
    # grid[4].set_xticks(xs, xticks)
    grid[4].set_xlabel('Position relative to exon start')
    grid[5].set_xlabel('Position relative to exon end')
    grid[6].set_xlabel('Position relative to exon start')
    grid[7].set_xlabel('Position relative to exon end')

    # plt.setp(grid[0].get_xticklabels(), visible=False)
    grid[0].tick_params(labelbottom=False)


    # ax[0, 0].imshow(mean_attn_ws_epochs[:, :140, 0], cmap='viridis')
    # ax[0, 1].imshow(mean_attn_ws_epochs[:, 140:, 0], cmap='viridis')
    # ax[0, 2].imshow(mean_attn_ws_epochs[:, :140, 1], cmap='viridis')
    # ax[0, 3].imshow(mean_attn_ws_epochs[:, 140:, 1], cmap='viridis')
    # ax[1, 0].imshow(mean_attn_ws_epochs[:, :140, 2], cmap='viridis')
    # ax[1, 1].imshow(mean_attn_ws_epochs[:, 140:, 2], cmap='viridis')
    # ax[1, 2].imshow(mean_attn_ws_epochs[:, :140, 2], cmap='viridis')
    # ax[1, 3].imshow(mean_attn_ws_epochs[:, 140:, 2], cmap='viridis')
    plt.savefig('attention_heatmap.png', dpi=300, bbox='tight')
    plt.show(dpi=300)

def meh():
    fig8 = plt.figure(constrained_layout=False)
    gs1 = fig8.add_gridspec(nrows=3, ncols=3, left=0.10, right=0.68, wspace=0.05)
    f8_ax1 = fig8.add_subplot(gs1[:-1, :])
    f8_ax2 = fig8.add_subplot(gs1[-1, :-1])
    f8_ax3 = fig8.add_subplot(gs1[-1, -1])
    plt.show()

def print_attn_sums(attn_ws):
    print(f'{sum(np.mean(attn_ws[:, :140], axis=0)):.2f}')
          # f' vs'
          # f' {sum(attn_ws[:, 140:]):.2f}')

heatmap()
# meh()

# attn_ws = np.load(f'attn_ws/attn_ws_50.npy')[:, :, 0]
# print_attn_sums(attn_ws)
# line_plot(attn_ws)

# attn_ws = np.load(f'attn_ws/attn_ws_50.npy')[:, :, 0]
# print_attn_sums(attn_ws)
# bar_chart(attn_ws)

# attn_ws_multi_heads = np.load(f'attn_ws_multi_distributed/attn_ws_110.npy')
# mean_attn_ws_multi_heads = np.mean(attn_ws_multi_heads, axis=-1)
# bar_chart(mean_attn_ws_multi_heads)


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




"""
Problem: doesn't really come out nicely if I normalize attention over whole 280 inputs / try to have 2 separate graphs
---> decision to not look at average but rather the 
"""
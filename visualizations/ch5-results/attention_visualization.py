import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from utils import one_hot_decode_seq_vanilla
from mpl_toolkits.axes_grid1 import ImageGrid
import matplotlib as mpl
from matplotlib.pyplot import gcf

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

def bar_chart_all(attn_ws):
    plt.style.use('seaborn')
    mean, std = np.mean(attn_ws, axis=0), np.std(attn_ws, axis=0)
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, sharey=True, figsize=(16, 4))
    xs = np.arange(-70, 70)
    xs2 = np.arange(0, 20)
    ax1.set_xlim(-70, 70)
    ax2.set_xlim(-70, 70)
    ax3.set_xlim(0, 20)
    ax4.set_xlim(0, 20)
    ax1.set_ylabel('Attention weight')

    ax1.bar(xs, mean[:140])#, yerr=stderr[:140])
    ax2.bar(xs, mean[140:])#, yerr=stderr[140:])
    ax3.bar(xs2, mean[60:80])#, yerr=stderr[:140])
    ax4.bar(xs2, mean[200:220])#, yerr=stderr[140:])

    ax1.xaxis.set_major_formatter(plt.NullFormatter())
    ax2.xaxis.set_major_formatter(plt.NullFormatter())
    ax3.xaxis.set_major_formatter(plt.NullFormatter())
    ax4.xaxis.set_major_formatter(plt.NullFormatter())

    exon_start, exon_end = 68/140, 70/140
    intron_end, intron_start = 67/140, 71/140
    ax1.annotate('intron', xy=(0, -0.025), xycoords='axes fraction', xytext=(0.25, -0.025),
                arrowprops=dict(arrowstyle="-", color='b'), ha='center', va='center')
    ax1.annotate('intron', xy=(intron_end+1.5/140, -0.025), xycoords='axes fraction', xytext=(0.25, -0.025),
                arrowprops=dict(arrowstyle="->", color='b'), ha='center', va='center')
    ax1.annotate('exon', xy=(exon_start-1.5/140, -0.025), xycoords='axes fraction', xytext=(0.75, -0.025),
                arrowprops=dict(arrowstyle="->", color='b'), ha='center', va='center')
    ax1.annotate('exon', xy=(1, -0.025), xycoords='axes fraction', xytext=(0.75, -0.025),
                arrowprops=dict(arrowstyle="-", color='b'), ha='center', va='center')

    ax2.annotate('exon', xy=(0, -0.025), xycoords='axes fraction', xytext=(0.25, -0.025),
                arrowprops=dict(arrowstyle="-", color='b'), ha='center', va='center')
    ax2.annotate('exon', xy=(exon_end+1.5/140, -0.025), xycoords='axes fraction', xytext=(0.25, -0.025),
                arrowprops=dict(arrowstyle="->", color='b'), ha='center', va='center')
    ax2.annotate('intron', xy=(intron_start-1.5/140, -0.025), xycoords='axes fraction', xytext=(0.75, -0.025),
                arrowprops=dict(arrowstyle="->", color='b'), ha='center', va='center')
    ax2.annotate('intron', xy=(1, -0.025), xycoords='axes fraction', xytext=(0.75, -0.025),
                arrowprops=dict(arrowstyle="-", color='b'), ha='center', va='center')
    # ax.grid(True)
    # fig.canvas.draw()

    exon_start, exon_end = 8/20, 10/20
    intron_end, intron_start = 7/20, 11/20
    y = -0.02
    ax3.annotate('intron', xy=(0, y), xycoords='axes fraction', xytext=((intron_end+0.0175)/2, y),
                arrowprops=dict(arrowstyle="-", color='b'), ha='center', va='center')
    ax3.annotate('intron', xy=(intron_end+0.0175, y), xycoords='axes fraction', xytext=((intron_end+0.0175)/2, y),
                arrowprops=dict(arrowstyle="->", color='b'), ha='center', va='center')
    ax3.annotate('exon', xy=(exon_start-0.045, y), xycoords='axes fraction', xytext=(0.75, y),
                arrowprops=dict(arrowstyle="->", color='b'), ha='center', va='center')
    ax3.annotate('exon', xy=(1, y), xycoords='axes fraction', xytext=(0.75, y),
                arrowprops=dict(arrowstyle="-", color='b'), ha='center', va='center')

    ax4.annotate('exon', xy=(0, y), xycoords='axes fraction', xytext=(0.25, y),
                arrowprops=dict(arrowstyle="-", color='b'), ha='center', va='center')
    ax4.annotate('exon', xy=(exon_end+1.5/140, y), xycoords='axes fraction', xytext=(0.25, y),
                arrowprops=dict(arrowstyle="->", color='b'), ha='center', va='center')
    ax4.annotate('intron', xy=(intron_start-0.055, y), xycoords='axes fraction', xytext=(0.75, y),
                arrowprops=dict(arrowstyle="->", color='b'), ha='center', va='center')
    ax4.annotate('intron', xy=(1, y), xycoords='axes fraction', xytext=(0.75, y),
                arrowprops=dict(arrowstyle="-", color='b'), ha='center', va='center')

    plt.tight_layout()

    plt.savefig('mean_attention_barchart_all.png', dpi=300, bbox='tight')

    plt.show(dpi=300)
    # print(attn_ws.shape)

def bar_chart_not_zoomed(attn_ws):
    plt.style.use('seaborn')
    mean, std = np.mean(attn_ws, axis=0), np.std(attn_ws, axis=0)
    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
    xs = np.arange(-70, 70)
    ax1.set_xlim(-70, 70)
    ax2.set_xlim(-70, 70)
    ax1.set_ylabel('Attention weight')

    ax1.bar(xs, mean[:140])#, yerr=stderr[:140])
    ax2.bar(xs, mean[140:])#, yerr=stderr[140:])

    ax1.xaxis.set_major_formatter(plt.NullFormatter())
    ax2.xaxis.set_major_formatter(plt.NullFormatter())

    # from when nucleotide extraction was still not perfect
    exon_start, exon_end = 68/140, 70/140
    intron_end, intron_start = 67/140, 71/140
    exon_start, exon_end = 0.5, 0.5
    intron_end, intron_start = 0.5, 0.5
    ax1.annotate('intron', xy=(0, -0.025), xycoords='axes fraction', xytext=(0.25, -0.025),
                arrowprops=dict(arrowstyle="-", color='b'), ha='center', va='center')
    ax1.annotate('intron', xy=(intron_end+1/140, -0.025), xycoords='axes fraction', xytext=(0.25, -0.025),
                arrowprops=dict(arrowstyle="->", color='b'), ha='center', va='center')
    ax1.annotate('exon', xy=(exon_start-1/140, -0.025), xycoords='axes fraction', xytext=(0.75, -0.025),
                arrowprops=dict(arrowstyle="->", color='b'), ha='center', va='center')
    ax1.annotate('exon', xy=(1, -0.025), xycoords='axes fraction', xytext=(0.75, -0.025),
                arrowprops=dict(arrowstyle="-", color='b'), ha='center', va='center')

    ax2.annotate('exon', xy=(0, -0.025), xycoords='axes fraction', xytext=(0.25, -0.025),
                arrowprops=dict(arrowstyle="-", color='b'), ha='center', va='center')
    ax2.annotate('exon', xy=(exon_end+1/140, -0.025), xycoords='axes fraction', xytext=(0.25, -0.025),
                arrowprops=dict(arrowstyle="->", color='b'), ha='center', va='center')
    ax2.annotate('intron', xy=(intron_start-1/140, -0.025), xycoords='axes fraction', xytext=(0.75, -0.025),
                arrowprops=dict(arrowstyle="->", color='b'), ha='center', va='center')
    ax2.annotate('intron', xy=(1, -0.025), xycoords='axes fraction', xytext=(0.75, -0.025),
                arrowprops=dict(arrowstyle="-", color='b'), ha='center', va='center')
    plt.tight_layout()

    plt.savefig('mean_attention_barchart_not_zoomed.png', dpi=300, bbox='tight')

    plt.show(dpi=300)
    # print(attn_ws.shape)

def bar_chart_zoomed(attn_ws, std=None):
    plt.style.use('seaborn')
    mean, std = np.mean(attn_ws, axis=0), np.std(attn_ws, axis=0)

    # print(f'Mean std dev: {np.mean(np.std(attn_ws, axis=0))}')
    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(10,5.0))
    xs = np.arange(0, 20)
    # ax1.set_xlabel('Position relative to exon start')
    # ax2.set_xlabel('Position relative to exon end')
    ax1.set_ylabel('Attention weight')
    ax1.set_xlim(-0.5, 19.5)
    ax2.set_xlim(-0.5, 19.5)

    # when nucleotide extraction was still biased
    # ax1.bar(xs, mean[67-10:67+10])
    # ax2.bar(xs, mean[211-10:211+10])
    if std is not None:
        ax1.bar(xs, mean[60:80], yerr=std[60:80])
        ax2.bar(xs, mean[200:220], yerr=std[200:220])
    else:
        ax1.bar(xs, mean[60:80],)
        ax2.bar(xs, mean[200:220],)
    ax1.set_ylim(0)
    ax2.set_ylim(0)
    ax1.xaxis.set_major_formatter(plt.NullFormatter())
    ax2.xaxis.set_major_formatter(plt.NullFormatter())
    # exon_start, exon_end = 8/20, 10/20
    # intron_end, intron_start = 7/20, 11/20
    y = -0.02
    ax1.annotate('intron', xy=(0, y), xycoords='axes fraction', xytext=(0.25, y),
                arrowprops=dict(arrowstyle="-", color='b'), ha='center', va='center')
    ax1.annotate('intron', xy=(0.5+0.005, y), xycoords='axes fraction', xytext=(0.25, y),
                arrowprops=dict(arrowstyle="->", color='b'), ha='center', va='center')
    ax1.annotate('exon', xy=(0.5-0.005, y), xycoords='axes fraction', xytext=(0.75, y),
                arrowprops=dict(arrowstyle="->", color='b'), ha='center', va='center')
    ax1.annotate('exon', xy=(1, y), xycoords='axes fraction', xytext=(0.75, y),
                arrowprops=dict(arrowstyle="-", color='b'), ha='center', va='center')

    ax2.annotate('exon', xy=(0, y), xycoords='axes fraction', xytext=(0.25, y),
                arrowprops=dict(arrowstyle="-", color='b'), ha='center', va='center')
    ax2.annotate('exon', xy=(0.5+0.005, y), xycoords='axes fraction', xytext=(0.25, y),
                arrowprops=dict(arrowstyle="->", color='b'), ha='center', va='center')
    ax2.annotate('intron', xy=(0.5-0.005, y), xycoords='axes fraction', xytext=(0.75, y),
                arrowprops=dict(arrowstyle="->", color='b'), ha='center', va='center')
    ax2.annotate('intron', xy=(1, y), xycoords='axes fraction', xytext=(0.75, y),
                arrowprops=dict(arrowstyle="-", color='b'), ha='center', va='center')
    # ax.grid(True)
    # fig.canvas.draw()

    plt.tight_layout()
    plt.savefig('mean_attention_barchart_zoomed.png', dpi=300, bbox='tight')

    plt.show(dpi=300)

def heatmap():
    # plt.style.use('seaborn')

    mean_attn_ws_epochs = []
    epochs = 100
    for i in range(epochs):
        attn_ws = np.load(f'attn_ws_multi_heads/attn_ws_cv_run_id=1_epoch={i+1}.npy')#[:,:140]

        mean = np.mean(attn_ws, axis=0)
        mean_attn_ws_epochs.append(mean)
    mean_attn_ws_epochs = np.concatenate(mean_attn_ws_epochs, axis=0).reshape(epochs, 280, 4)

    im1, im2 = mean_attn_ws_epochs[:, :, 0], mean_attn_ws_epochs[:, :, 0]
    im3, im4 = mean_attn_ws_epochs[:, :, 1], mean_attn_ws_epochs[:, :, 1]
    im5, im6 = mean_attn_ws_epochs[:, :, 2], mean_attn_ws_epochs[:, :, 2]
    im7, im8 = mean_attn_ws_epochs[:, :, 3], mean_attn_ws_epochs[:, :, 3]

    fig = plt.figure(figsize=(16., 7.))
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                     nrows_ncols=(2, 4),  # creates 2x2 grid of axes
                     axes_pad=0.3,  # pad between axes in inch.
                     )

    show_first_140_nt = True
    cmap = 'inferno'
    for i, (ax, im) in enumerate(zip(grid, [im1, im2, im3, im4, im5, im6, im7, im8])):
        # Iterating over the grid returns the Axes.
        # if i == 0: continue
        if show_first_140_nt:
            imshow = ax.imshow(im, extent=[-70, 210, epochs, 0], cmap=cmap)
            ax.set_xlim(-70, 70)
        else:
            imshow = ax.imshow(im, extent=[-210, 70, epochs, 0], cmap=cmap)
            ax.set_xlim(-70, 70)

        show_first_140_nt = not show_first_140_nt
        # if i==7:
        #     plt.colorbar(imshow)

    # grid[0].set_xticks([], [])

    grid[0].set_title('Start sequence')
    grid[1].set_title('End sequence')
    grid[2].set_title('Start sequence')
    grid[3].set_title('End sequence')

    grid[0].set_ylabel('Epochs')
    grid[4].set_ylabel('Epochs')

    xs = np.arange(-70, 70)
    xticks = np.linspace(-70, 70, 5)
    # grid[4].set_xticks(xs, xticks)
    line_color = 'darkred'
    grid[0].plot([0.5171, 0.5171], [0.05, 0.95], color=line_color, lw=3.5,
             transform=gcf().transFigure, clip_on=False)
    # grid[0].plot([0., 0.], [0, 1], color=line_color, lw=3.5,
    #          transform=gcf().transFigure, clip_on=False)
    grid[0].plot([0.025, 0.993], [0.495, 0.495], color=line_color, lw=3,
             transform=gcf().transFigure, clip_on=False)

    # 1.5/140
    exon_start, exon_end = 69/140, 71/140
    intron_end, intron_start = 68/140, 72/140
    exon_start, exon_end = 0.5, 0.5
    intron_end, intron_start = 0.5, 0.5

    for i in range(8): grid[i].set_xticks([])
    y = -0.0375
    for i in range(0, 8, 2):
        grid[i].annotate('intron', xy=(0, y), xycoords='axes fraction', xytext=(0.25, y),
                    arrowprops=dict(arrowstyle="-", color='black'), ha='center', va='center')
        grid[i].annotate('intron', xy=(intron_end+1.5/140, y), xycoords='axes fraction', xytext=(0.25, y),
                    arrowprops=dict(arrowstyle="->", color='black'), ha='center', va='center')
        grid[i].annotate('exon', xy=(exon_start-1.5/140, y), xycoords='axes fraction', xytext=(0.75, y),
                    arrowprops=dict(arrowstyle="->", color='black'), ha='center', va='center')
        grid[i].annotate('exon', xy=(1, y), xycoords='axes fraction', xytext=(0.75, y),
                    arrowprops=dict(arrowstyle="-", color='black'), ha='center', va='center')
    for i in range(1, 8, 2):
        grid[i].annotate('exon', xy=(0, y), xycoords='axes fraction', xytext=(0.25, y),
                     arrowprops=dict(arrowstyle="-", color='black'), ha='center', va='center')
        grid[i].annotate('exon', xy=(exon_end + 1.5 / 140, y), xycoords='axes fraction', xytext=(0.25, y),
                     arrowprops=dict(arrowstyle="->", color='black'), ha='center', va='center')
        grid[i].annotate('intron', xy=(intron_start - 1.5 / 140, y), xycoords='axes fraction', xytext=(0.75, y),
                     arrowprops=dict(arrowstyle="->", color='black'), ha='center', va='center')
        grid[i].annotate('intron', xy=(1, y), xycoords='axes fraction', xytext=(0.75, y),
                     arrowprops=dict(arrowstyle="-", color='black'), ha='center', va='center')
    grid[0].patch.set_facecolor('black')
    # grid[0].patch.set_alpha(0.7)
    grid[0].set_facecolor('black')
    # plt.setp(grid[0].get_xticklabels(), visible=False)
    grid[0].tick_params(labelbottom=False)

    plt.tight_layout()
    plt.savefig('attention_heatmap.png', dpi=300, bbox='tight')
    plt.show(dpi=300)

heatmap()

def heatmap_rectangle(attn_ws):
    h = 10
    mean, std = np.mean(attn_ws[:, 200:220], axis=0), np.std(attn_ws, axis=0)
    N = mean.shape[0]

    cmap = plt.get_cmap(name='viridis')
    norm = mpl.colors.Normalize(vmin=min(mean), vmax=max(mean))

    fig=plt.figure()
    ax=fig.add_subplot(111)
    plt.axis('scaled')
    ax.set_xlim([ 0, N])
    ax.set_ylim([-h//2, h//2])

    for i in range(N):
        cmap_value = norm(mean[i])
        rect = plt.Rectangle((i, -h//2), width=1, height=h, facecolor=cmap(cmap_value))
        ax.add_artist(rect)
    ax.set_yticks([])
    plt.show()

def load_all_runs():
    ws = [np.load(f'attn_ws_multi_heads/attn_ws_cv_run_id={i}.npy') for i in range(9)]
    return np.array(ws)

def average_over_runs(ws):
    return np.mean(ws, axis=0), np.std(ws, axis=0)

def average_over_heads(attn_ws):
    return np.mean(attn_ws, axis=-1)
attn_ws_multi_heads = np.load(f'attn_ws_multi_heads/attn_ws_cv_run_id=2.npy')

attn = load_all_runs()
mean_attn = average_over_heads(attn)
mean_attn, std_attn = average_over_runs(mean_attn)

# attn_ws_multi_heads = np.load(f'attn_ws_multi_heads/attn_ws_cv_run_id=1_epoch=152.npy')

# mean_attn_ws_multi_heads = np.mean(attn_ws_multi_heads, axis=-1)


# bar_chart_not_zoomed(mean_attn)

# bar_chart_zoomed(mean_attn, std_attn)

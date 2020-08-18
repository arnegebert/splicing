import numpy as np
import matplotlib.pyplot as plt
sample = 2
attn_ws = np.load(f'attn_ws/attn_ws_50.npy')
attn_ws = attn_ws[sample, :, 0]

# fig, axs = plt.subplots(1, 2, sharey=True)
# axs[0].plot(attn_ws[:140])
# axs[1].plot(attn_ws[140:])
fig, ax = plt.subplots()
x_ticks = [i for i in range(-140, 141) if i]#np.arange(-140, -1, -1)
ax.set_xlim(-140, 140)
ax.plot(x_ticks, attn_ws, )
print(x_ticks)
ax.annotate('Sequence change',
            xy=(0, 0),
            xytext=(0, 0.001),
            arrowprops = dict(facecolor='black', shrink=0.05))
plt.show()
print(attn_ws.shape)
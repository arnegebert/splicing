import gensim.models

xxx = gensim.models.Word2Vec.load('../../model/w2v-full-5epochs')
print('x')

from sklearn.manifold import TSNE                   # final reduction
import numpy as np                                  # array handling


def reduce_dimensions(model):
    num_dimensions = 2  # final num dimensions (2D, 3D, etc)

    vectors = [] # positions in vector space
    labels = [] # keep track of words to label our data again later
    for word in model.wv.vocab:
        vectors.append(model.wv[word])
        labels.append(word)

    # convert both lists into numpy vectors for reduction
    vectors = np.asarray(vectors)
    labels = np.asarray(labels)

    # reduce using t-SNE
    vectors = np.asarray(vectors)
    tsne = TSNE(n_components=num_dimensions, random_state=1)
    vectors = tsne.fit_transform(vectors)

    x_vals = [v[0] for v in vectors]
    y_vals = [v[1] for v in vectors]
    return x_vals, y_vals, labels

model = xxx
x_vals, y_vals, labels = reduce_dimensions(model)

def plot_with_plotly(x_vals, y_vals, labels, plot_in_notebook=True):
    # from plotly.offline import init_notebook_mode, iplot, plot
    # import plotly.graph_objs as go
    #
    # trace = go.Scatter(x=x_vals, y=y_vals, mode='text', text=labels)
    # data = [trace]
    #
    # if plot_in_notebook:
    #     init_notebook_mode(connected=True)
    #     iplot(data, filename='word-embedding-plot')
    # else:
    #     plot(data, filename='word-embedding-plot.html')
    print('no')


def plot_with_matplotlib(x_vals, y_vals, labels):
    import matplotlib.pyplot as plt
    import random

    random.seed(0)

    plt.figure(figsize=(12, 12))
    plt.scatter(x_vals, y_vals)

    #
    # Label randomly subsampled 25 data points
    #
    indices = list(range(len(labels)))
    selected_indices = random.sample(indices, 64)
    for i in selected_indices:
        plt.annotate(labels[i], (x_vals[i], y_vals[i]))
    plt.savefig('tSNE-w2v.png')
    plt.show()

# try:
#     get_ipython()
# except Exception:
#     plot_function = plot_with_matplotlib
# else:
#     plot_function = plot_with_plotly


plot_with_matplotlib(x_vals, y_vals, labels)

def infinite_sequence():
    num = 0
    for x in range(10):
        yield num
        num += 1

x = infinite_sequence()
for e in x:
    print(e)
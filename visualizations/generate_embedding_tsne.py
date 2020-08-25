import gensim.models

model_name = 'd2v'
# d2v: 7.5 I like the best, 25 previous favorite
# w2v: 12.5 I like the best, 10 also decent
perplexity = 7.5
embedding_model = gensim.models.Word2Vec.load(f'../model/{model_name}-full-5epochs')
save_to = f'tSNE-{model_name}.png'
print('x')

from sklearn.manifold import TSNE                   # final reduction
import numpy as np                                  # array handling


def codon_to_amino_acid(codon):
    mapping = {
        'ATA': 'I', 'ATC': 'I', 'ATT': 'I', 'ATG': 'M',
        'ACA': 'T', 'ACC': 'T', 'ACG': 'T', 'ACT': 'T',
        'AAC': 'N', 'AAT': 'N', 'AAA': 'K', 'AAG': 'K',
        'AGC': 'S', 'AGT': 'S', 'AGA': 'R', 'AGG': 'R',
        'CTA': 'L', 'CTC': 'L', 'CTG': 'L', 'CTT': 'L',
        'CCA': 'P', 'CCC': 'P', 'CCG': 'P', 'CCT': 'P',
        'CAC': 'H', 'CAT': 'H', 'CAA': 'Q', 'CAG': 'Q',
        'CGA': 'R', 'CGC': 'R', 'CGG': 'R', 'CGT': 'R',
        'GTA': 'V', 'GTC': 'V', 'GTG': 'V', 'GTT': 'V',
        'GCA': 'A', 'GCC': 'A', 'GCG': 'A', 'GCT': 'A',
        'GAC': 'D', 'GAT': 'D', 'GAA': 'E', 'GAG': 'E',
        'GGA': 'G', 'GGC': 'G', 'GGG': 'G', 'GGT': 'G',
        'TCA': 'S', 'TCC': 'S', 'TCG': 'S', 'TCT': 'S',
        'TTC': 'F', 'TTT': 'F', 'TTA': 'L', 'TTG': 'L',
        'TAC': 'Y', 'TAT': 'Y', 'TAA': '_', 'TAG': '_',
        'TGC': 'C', 'TGT': 'C', 'TGA': '_', 'TGG': 'W',
    }
    return mapping[codon]

def one_l_codon_to_three_l_codon(one_letter_codon_abbrv):
    mapping = {
        'F' : 'Phe',
        'L' : 'Leu',
        'S': 'Ser',
        'Y': 'Tyr',
        'X': 'Ter',
        'C': 'Cys',
        'W': 'Trp',
        'P': 'Pro',
        'H': 'His',
        'Q': 'Gln',
        'R': 'Arg',
        'I': 'Ile',
        'M': 'Met',
        'T': 'Thr',
        'N': 'Asn',
        'K': 'Lys',
        'V': 'Val',
        'A': 'Ala',
        'D': 'Asp',
        'E': 'Glu',
        'G': 'Gly',
        'B': 'Asp/Asn',
        'Z': 'Glu/Gln',
        '_': 'Ter'
    }
    return mapping[one_letter_codon_abbrv]


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

    tsne = TSNE(n_components=num_dimensions, random_state=1, perplexity=perplexity)
    vectors = tsne.fit_transform(vectors)

    x_vals = [v[0] for v in vectors]
    y_vals = [v[1] for v in vectors]
    return x_vals, y_vals, labels

model = embedding_model
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
    # Label randomly subsampled 64 data points
    #
    indices = list(range(len(labels)))
    selected_indices = random.sample(indices, 64)
    for i in selected_indices:
        # plt.annotate(codon_to_amino_acid(labels[i]), (x_vals[i], y_vals[i]))
        label_name = codon_to_amino_acid(labels[i])
        label_name = one_l_codon_to_three_l_codon(label_name)
        xy = (x_vals[i], y_vals[i])
        xytext = (x_vals[i]-3.5, y_vals[i]+2)
        plt.annotate(label_name, xy=(x_vals[i], y_vals[i]),xytext=xytext)
        # plt.annotate(labels[i], xy=(x_vals[i], y_vals[i]), xytext=xytext)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig(save_to, bbox_inches='tight', dpi=300)
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




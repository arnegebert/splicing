import gensim.models
from collections import defaultdict
import matplotlib.cm as cm
import matplotlib.pyplot as plt

model_name = 'd2v'
# d2v: 7.5 I like the best, 25 previous favorite
# w2v: 12.5 I like the best, 10 also decent
perplexity = 7.5
embedding_model = gensim.models.Word2Vec.load(f'../../model/{model_name}-full-5epochs')
display_codon = False
codon_suffix = '' if display_codon else '_bases'
save_to = f'tSNE-{model_name}{codon_suffix}.png'

from sklearn.manifold import TSNE                   # final reduction
import numpy as np                                  # array handling

codon_to_amino_acid_mapping = {
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

def codon_to_amino_acid(codon):
    return codon_to_amino_acid_mapping[codon]

def one_letter_aacid_to_three_letter_aacid_abbrv(one_letter_codon_abbrv):
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

def plot_with_matplotlib2(x_vals, y_vals, labels):
    plt.figure(figsize=(9, 9))
    plt.scatter(x_vals, y_vals)
    indices = list(range(len(labels)))
    for i in indices:
        if not display_codon:
            label_name = labels[i]
        else:
            label_name = codon_to_amino_acid(labels[i])
            label_name = one_letter_aacid_to_three_letter_aacid_abbrv(label_name)
        xytext = (x_vals[i]-3.5, y_vals[i]+2)
        plt.annotate(label_name, xy=(x_vals[i], y_vals[i]),xytext=xytext)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig(save_to, bbox_inches='tight', dpi=300)
    plt.show()

# plot_with_matplotlib2(x_vals, y_vals, labels)

def plot_with_matplotlib(x_vals, y_vals, labels):

    plt.figure(figsize=(11, 11))

    # getting mapping from amino acid -> all codons
    # iterate through acids and plot all codons with specific color
    # same annotation afterwards

    amino_acid_to_codons = dict()
    # reverse dictionary
    for codon, amino_acid in codon_to_amino_acid_mapping.items():
        idx = np.where(labels==codon)[0][0]
        amino_acid_to_codons.setdefault(amino_acid, list()).append((codon, idx))

    # color map
    x = np.arange(21)
    # 21 different amino acids
    ys = [i + x + (i * x) ** 2 for i in range(21)]
    # colors = cm.rainbow(np.linspace(0, 1, len(ys)))
    colors = plt.cm.get_cmap(lut=21, name='tab20c')

    for j, (amino_acid, codon_idx_tuples) in enumerate(amino_acid_to_codons.items()):
        # get x/y values based on indexes
        amino_acid_x_vals = [x_vals[idx] for (codon, idx) in codon_idx_tuples]
        amino_acid_y_vals = [y_vals[idx] for (codon, idx) in codon_idx_tuples]
        amino_acid = one_letter_aacid_to_three_letter_aacid_abbrv(amino_acid)
        plt.scatter(amino_acid_x_vals, amino_acid_y_vals, color=colors(j), label=amino_acid)

    for i, label in enumerate(labels):
        if not display_codon:
            label_name = label
        else:
            label_name = codon_to_amino_acid(label)
            label_name = one_letter_aacid_to_three_letter_aacid_abbrv(label_name)
        xytext = (x_vals[i]-3.5, y_vals[i]+2)
        plt.annotate(label_name, xy=(x_vals[i], y_vals[i]),xytext=xytext)

    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.savefig('tsne_d2v_embeddings.png', bbox_inches='tight', dpi=300)
    plt.show()

plot_with_matplotlib(x_vals, y_vals, labels)

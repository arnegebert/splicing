import math
import gensim.models
import tempfile
import time

sentence_len = 30000


def split_into_sentences(text, n):
    return [text[i * n:(i + 1) * n] for i in range(0, math.ceil(len(text) / n))]


def split_into_3_mers(sentence):
    words = []
    for i in range(1, len(sentence) - 1):
        words.append(sentence[i - 1:i + 2])
    return words


class HumanGenomeCorpus(object):
    def __init__(self):
        self.epoch = 1

    def __iter__(self):
        print(f'Starting epoch {self.epoch}')
        self.epoch += 1
        for i in range(1, 2):
            with open(f'../../data/chromosomes/chr{i}.fa') as f:
                print(f'Loaded chromosome {i}')
                seq = f.read().replace('\n', '')
                if i < 10:
                    seq = seq[5:].replace('N', '').replace('n', '').upper()
                else:
                    seq = seq[6:].replace('N', '').replace('n', '').upper()
                seq_sentences = split_into_sentences(seq, sentence_len)
                print(f'Split chromosome {i} into sentences')
                for sentence in seq_sentences:
                    yield split_into_3_mers(sentence)


sentences = HumanGenomeCorpus()
print('Starting training')
start = time.time()
model = gensim.models.Word2Vec(sentences=sentences, size=100, min_count=5, window=5,
                                            sg=0, workers=8, iter=1, negative=5)
end = time.time()
# 1 worker: 220 s
# 5 workes: 160 s for 1 epoch on chrom 1 x_x
# 8 workers: 155 s
print(f'Time for training {end-start}')
# model.build_vocab(sentences)
# model.train(sentences=sentences, epochs=5)
# print('Training finished')

model.save('embedding-model')

print('Model saved')
# with tempfile.NamedTemporaryFile(prefix='gensim-model-', delete=False) as tmp:
#     temporary_filepath = tmp.name
#     model.save(temporary_filepath)
#     #
#     # The model is now safely stored in the filepath.
#     # You can copy it to other machines, share it with others, etc.
#     #
#     # To load a saved model:
#     #
#     # new_model = gensim.models.Word2Vec.load(temporary_filepath)



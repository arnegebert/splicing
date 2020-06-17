import math
import gensim.models
import tempfile
import time

sentence_len = 30000
continue_training = False
epochs = 5

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
        print('-------------------------------')
        print(f'Starting epoch {self.epoch}')
        self.epoch += 1
        for i in range(1, 23):
            with open(f'../data/chromosomes/chr{i}.fa') as f:
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

    # nvm, takes 5.5 GIG for chromosome 21 and 22 alone ;______;
    def get_all_sentences(self):
        print('-------------------------------')
        print(f'Starting epoch {self.epoch}')
        self.epoch += 1
        all = []
        for i in range(21, 23):
            with open(f'../data/chromosomes/chr{i}.fa') as f:
                print(f'Loaded chromosome {i}')
                seq = f.read().replace('\n', '')
                if i < 10:
                    seq = seq[5:].replace('N', '').replace('n', '').upper()
                else:
                    seq = seq[6:].replace('N', '').replace('n', '').upper()
                seq_sentences = split_into_sentences(seq, sentence_len)
                print(f'Split chromosome {i} into sentences')
                # processed = []
                # for sentence in seq_sentences:
                #     processed.append(split_into_3_mers(sentence))
            all.extend(map(split_into_3_mers, seq_sentences))
        return all


# w o w
# takes 11 s for 19-23 if I give a list
# takes 49 s for 19-23 if I use a generator

sentences = HumanGenomeCorpus()#.get_all_sentences()
print('Starting training')
start = time.time()
if not continue_training:
    model = gensim.models.Word2Vec(sentences=sentences, size=100, min_count=5, window=5,
                                            sg=0, workers=8, iter=epochs-1, negative=5)
else:
    model = gensim.models.Word2Vec.load('w2v-full-5epochs')
    model.train(sentences=sentences, epochs=2, total_examples=model.corpus_count, word_count=0)

end = time.time()
# 5 epochs, full = 1.5 h
# 20 epochs, full = 6 h
print(f'Time for training {end-start}')
# model.build_vocab(sentences)
# model.train(sentences=sentences, epochs=5)
# print('Training finished')

model.save('w2v-full-5epochs-x')

print('Model saved')

def print_voc(model):
    for i, word in enumerate(model.wv.vocab):
        if i == 10:
            break
        print(word)

print_voc(model)

import math
import gensim.models
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
        t = time.localtime()
        current_time = time.strftime("%H:%M:%S", t)
        print(current_time)
        self.epoch += 1
        doc_id = 0
        for i in range(1, 23):
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
                yield gensim.models.doc2vec.TaggedDocument(split_into_3_mers(sentence), [doc_id])
                doc_id += 1
        print(f'Number of documents: {doc_id+1}')

# w o w
# takes 11 s for 19-23 if I give a list
# takes 49 s for 19-23 if I use a generator

sentences = HumanGenomeCorpus()
print('Starting training')
# gen = sentences.__iter__()
# x1 = next(gen)
# x2 = next(gen)
# x3 = next(gen)
start = time.time()
if not continue_training:
    model = gensim.models.Doc2Vec(documents=sentences, vector_size=100, min_count=3, window=5,
                                            dm=1, workers=8, epochs=epochs-1, negative=5)
else:
    model = gensim.models.Word2Vec.load('d2v-full-5epochs')
    model.train(sentences=sentences, epochs=2, total_examples=model.corpus_count, word_count=0)

end = time.time()
# 5 epochs, full = 1.5 h
# 20 epochs, full = 6 h
print(f'Time for training {end-start}')
# model.build_vocab(sentences)
# model.train(sentences=sentences, epochs=5)
# print('Training finished')

model.save('d2v-full-5epochs')

print('Model saved')

def print_voc(model):
    for i, word in enumerate(model.wv.vocab):
        if i == 10:
            break
        print(word)

print_voc(model)

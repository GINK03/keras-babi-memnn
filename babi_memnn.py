'''Trains a memory network on the bAbI dataset.
References:
- Jason Weston, Antoine Bordes, Sumit Chopra, Tomas Mikolov, Alexander M. Rush,
  "Towards AI-Complete Question Answering: A Set of Prerequisite Toy Tasks",
  http://arxiv.org/abs/1502.05698
- Sainbayar Sukhbaatar, Arthur Szlam, Jason Weston, Rob Fergus,
  "End-To-End Memory Networks",
  http://arxiv.org/abs/1503.08895
Reaches 98.6% accuracy on task 'single_supporting_fact_10k' after 120 epochs.
Time per epoch: 3s on CPU (core i7).
'''

from __future__ import print_function
from keras.models import Sequential, load_model
from keras.layers.embeddings import Embedding
from keras.layers import Activation, Dense, Merge, Permute, Dropout
from keras.layers import LSTM
from keras.utils.data_utils import get_file
from keras.preprocessing.sequence import pad_sequences
from functools import reduce
import tarfile
import numpy as np
import re
import sys
import pickle

import Model
def tokenize(sent):
    '''Return the tokens of a sentence including punctuation.
    >>> tokenize('Bob dropped the apple. Where is the apple?')
    ['Bob', 'dropped', 'the', 'apple', '.', 'Where', 'is', 'the', 'apple', '?']
    '''
    return [x.strip() for x in re.split('(\W+)?', sent) if x.strip()]


def parse_stories(lines, only_supporting=False):
    '''Parse stories provided in the bAbi tasks format
    If only_supporting is true, only the sentences that support the answer are kept.
    '''
    """
    bAbiタスクの要素の一つの例。こんな感じになってる
    ([['John', 'moved', 'to', 'the', 'bathroom', '.'], ['Daniel', 'went', 'to', 'the', 'kitchen', '.'], ['Sandra', 'travelled', 'to', 'the', 'kitchen', '.'], ['Mary', 'travelled', 'to', 'the', 'hallway', '.'], ['Sandra', 'went', 'back', 'to', 'the', 'bathroom', '.'], ['John', 'went', 'back', 'to', 'the', 'kitchen', '.'], ['Daniel', 'went', 'back', 'to', 'the', 'office', '.'], ['Daniel', 'journeyed', 'to', 'the', 'bathroom', '.'], ['John', 'went', 'back', 'to', 'the', 'office', '.'], ['Mary', 'travelled', 'to', 'the', 'bedroom', '.']], ['Where', 'is', 'John', '?'], 'office')
    """
    data = []
    story = []
    for line in lines:
        line = line.strip()
        nid, line = line.split(' ', 1)
        nid = int(nid)
        if nid == 1:
            story = []
        if '\t' in line:
            q, a, supporting = line.split('\t')
            q = tokenize(q)
            substory = None
            if only_supporting:
                # Only select the related substory
                supporting = map(int, supporting.split())
                substory = [story[i - 1] for i in supporting]
            else:
                # Provide all the substories
                substory = [x for x in story if x]
            data.append((substory, q, a))
            story.append('')
        else:
            sent = tokenize(line)
            story.append(sent)
    #for d in data:
    #  print(d)
    #sys.exit()
    return data


def get_stories(f, only_supporting=False, max_length=None):
    '''Given a file name, read the file, retrieve the stories, and then convert the sentences into a single story.
    If max_length is supplied, any stories longer than max_length tokens will be discarded.
    '''
    """
    すべての状況説明センテンスが一つに結合されて、質問と回答が記される
    (['Sandra', 'went', 'to', 'the', 'bathroom', '.', 'John', 'moved', 'to', 'the', 'bathroom', '.'], ['Where', 'is', 'John', '?'], 'bathroom')
    """
    data = parse_stories(f.readlines(), only_supporting=only_supporting)
    flatten = lambda data: reduce(lambda x, y: x + y, data)
    data = [(flatten(story), q, answer) for story, q, answer in data if not max_length or len(flatten(story)) < max_length]
    return data

def get_stories2(name, only_supporting=False, max_length=None):
    f = open(name, 'rb')
    return pickle.loads(f.read())

def vectorize_stories(data, word_idx, story_maxlen, query_maxlen):
    """
    X;  [[4, 16, 19, 18, 9, 1, 3, 21, 19, 18, 15, 1.... ]
    Xq; [[7, 13, 3, 2], ... ]
    Y; np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]) <- one-hot? , one-hotっぽい
    """
    X = []
    Xq = []
    Y = []
    for story, query, answer in data:
        x = [word_idx[w] for w in story]
        xq = [word_idx[w] for w in query]
        y = np.zeros(len(word_idx) + 1)  # let's not forget that index 0 is reserved
        y[word_idx[answer]] = 1
        X.append(x)
        Xq.append(xq)
        Y.append(y)
    return (pad_sequences(X, maxlen=story_maxlen),
            pad_sequences(Xq, maxlen=query_maxlen), np.array(Y))

challenge = 'tasks_1-20_v1-2/en-10k/qa1_single-supporting-fact_{}.txt'
challenge = 'tasks_1-20_v1-2/ja-10k/ja.single-supporting-fact_{}.pkl'
print('Extracting stories for the challenge:', challenge)
stories_getter = get_stories2
train_stories = stories_getter(challenge.format('train'))
test_stories  = stories_getter(challenge.format('test'))


vocab = sorted(reduce(lambda x, y: x | y, (set(story + q + [answer]) for story, q, answer in train_stories + test_stories)))
for i, v in enumerate(vocab):
  print(i,v)
# Reserve 0 for masking via pad_sequences
vocab_size = len(vocab) + 1
story_maxlen = max(map(len, (x for x, _, _ in train_stories + test_stories)))
query_maxlen = max(map(len, (x for _, x, _ in train_stories + test_stories)))

print('-')
print('Vocab size:', vocab_size, 'unique words')
print('Story max length:', story_maxlen, 'words')
print('Query max length:', query_maxlen, 'words')
print('Number of training stories:', len(train_stories))
print('Number of test stories:', len(test_stories))
print('-')
print('Here\'s what a "story" tuple looks like (input, query, answer):')
print(train_stories[0])
print('-')
print('Vectorizing the word sequences...')

word_idx = dict((c, i + 1) for i, c in enumerate(vocab))
iword_idx = dict((i + 1, c) for i, c in enumerate(vocab))
inputs_train, queries_train, answers_train = vectorize_stories(train_stories, word_idx, story_maxlen, query_maxlen)
inputs_test, queries_test, answers_test = vectorize_stories(test_stories, word_idx, story_maxlen, query_maxlen)

print('-')
print('inputs: integer tensor of shape (samples, max_length)')
print('inputs_train shape:', inputs_train.shape)
print('inputs_test shape:', inputs_test.shape)
print('-')
print('queries: integer tensor of shape (samples, max_length)')
print('queries_train shape:', queries_train.shape)
print('queries_test shape:', queries_test.shape)
print('-')
print('answers: binary (1 or 0) tensor of shape (samples, vocab_size)')
print('answers_train shape:', answers_train.shape)
print('answers_test shape:', answers_test.shape)
print('-')
print('Compiling...')


def train():
  mode = Model.MODE.NORMAL
  # Note: you could use a Graph model to avoid repeat the input twice
  model = Model.Model()(mode, vocab_size=vocab_size, query_maxlen=query_maxlen, story_maxlen=story_maxlen)
  model.fit([inputs_train, queries_train, inputs_train], answers_train,
           batch_size=32,
           nb_epoch=200,
           validation_data=([inputs_test, queries_test, inputs_test], answers_test))
  model.save_weights('keras.weights')

def pred():
  JOIN_TARGET = ''
  mode = Model.MODE.NORMAL
  model = Model.Model()(mode, vocab_size=vocab_size, query_maxlen=query_maxlen, story_maxlen=story_maxlen)
  model.load_weights('keras.weights')
  preds = model.predict([inputs_test, queries_test, inputs_test])
  for i, (input, query, pred) in enumerate(zip(inputs_test, queries_test, preds)):
    ans_idx = np.argmax(pred)
    print("dataset No.%d"%i)
    output = JOIN_TARGET.join(list(map(lambda x:str(iword_idx.get(x)), input.tolist() ) ) )
    output = output.replace('None', '').replace(' .', '.')
    print(output)
    output = JOIN_TARGET.join(list(map(lambda x:str(iword_idx.get(x)), query.tolist() ) ) ) 
    output = output.replace('None', '').replace(' .', '.')
    print('Q:%s'%output)
    print("ANS:%s"%iword_idx[ans_idx])

def main():
  if '--train' in sys.argv:
    train()
  if '--pred' in sys.argv:
    pred()
    
if __name__ == '__main__':
  main()

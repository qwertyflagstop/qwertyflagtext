from keras.utils.np_utils import to_categorical
from math import *
import numpy as np
import string

class TextFile(object):

    def __init__(self, text_fp):
        super(TextFile, self).__init__()
        self.fp = text_fp
        self.token_map = {}
        self.token_list = None
        self.offset = 0
        self.one_hot_indexes = None
        self.vocab_size = 0
        self.parse()


    def parse(self):
        with open(self.fp,'r') as fp:
            master_string = fp.read()
            master_string = [x for x in master_string if x in string.printable]
            s = sorted(list(set(master_string))) #important!!!
            index = 0
            tokes = []
            for c in s:
                self.token_map[c] = index
                tokes.append(c)
                index += 1
            self.token_list = np.array(tokes)
            self.vocab_size = self.token_list.shape[0]
            self.one_hot_indexes = np.zeros((len(master_string),), np.uint8)
            for i, c in enumerate(master_string):
                self.one_hot_indexes[i] = self.token_map[c]
            print('{} tokens with a vocab size of {}'.format(self.one_hot_indexes.shape[0], len(s)))

    def get_epochs(self):
        return  int(floor(self.offset/self.one_hot_indexes.shape[0]))

    def get_mini_batch(self, batch_size, input_steps):
        j = self.offset % (self.one_hot_indexes.shape[0]-(batch_size*input_steps+input_steps))
        self.offset = (batch_size*input_steps + self.offset)
        chars_seqs_in = np.zeros((batch_size,input_steps))
        chars_seqs_out = np.zeros((batch_size,input_steps,self.vocab_size))
        for i in np.arange(0,batch_size):
            chars_seqs_in[i] = self.one_hot_indexes[j + i*input_steps:j + i*input_steps + input_steps]
            chars_seqs_out[i] =  to_categorical(self.one_hot_indexes[j + i*input_steps+1:j+i*input_steps+1+input_steps],self.vocab_size)
        return (chars_seqs_in, chars_seqs_out)

    def hot_to_char(self, hot):
        return self.token_list[np.argmax(hot)]

    def indexes_to_string(self, hots):
        return ''.join([self.token_list[x] for x in hots])

if __name__ == '__main__':
    tf = TextFile('trainTexts/tinder.txt')
    X,Y = tf.get_mini_batch(32, 25, 1, in_order=False)
    X, Y = tf.get_mini_batch(32, 25, 1, in_order=False)
    for i in np.arange(0,Y.shape[0]):
        print(''.join([tf.token_list[int(i)] for i in X[i]]))
        print(tf.hot_to_char(Y[i]))

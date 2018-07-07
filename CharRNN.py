from keras import Model
from keras.utils.generic_utils import Progbar
from keras.models import Sequential
from keras.layers import Dense, InputLayer, Activation, Input, Concatenate, Reshape, Embedding
from keras.layers import GRU, TimeDistributed, Bidirectional, ConvLSTM2D, Dropout, BatchNormalization
from Utils import ResetStateCallback
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from TextProcesor import TextFile
import os
from Utils import Logger

class CharRNN(object):

    def __init__(self, name, vocab_size, batch_size, sequence_length=None, temperature=1.0):
        super(CharRNN, self).__init__()
        self.is_stateful = True
        self.name = name
        self.vocab_size = vocab_size
        self.batch_size = batch_size
        self.sequence_length = batch_size if sequence_length is None else sequence_length
        self.model = self.make_train_model(self.batch_size)
        self.pred_model = self.make_pred_model(1)
        self.temp = temperature

    def make_train_model(self, batch_size):
        model = Sequential()
        #model.add(InputLayer(batch_input_shape=(batch_size, self.sequence_length, self.vocab_size)))
        model.add(Embedding(batch_input_shape=(batch_size, self.sequence_length), input_dim=self.vocab_size, output_dim=256))
        for i in np.arange(0, 3):
            model.add(GRU(256, return_sequences=True, stateful=False, recurrent_dropout=0.3))
            model.add(Dropout(0.3))
        model.add(TimeDistributed(Dense(256, activation='relu')))
        model.add(TimeDistributed(Dense(self.vocab_size)))
        model.add(Activation('softmax'))
        return model

    def make_pred_model(self, batch_size):
        model = Sequential()
        #model.add(InputLayer(batch_input_shape=(batch_size, self.sequence_length, self.vocab_size)))
        model.add(Embedding(batch_input_shape=(batch_size, 1), input_dim=self.vocab_size, output_dim=256))
        for i in np.arange(0, 3):
            model.add(GRU(256, return_sequences=True, stateful=True))
        model.add(TimeDistributed(Dense(256, activation='relu')))
        model.add(TimeDistributed(Dense(self.vocab_size)))
        model.add(Activation('softmax'))
        return model

    def train_lstm(self, tf, logger):
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
        batches = 0
        epochs = -1
        while True:
            X,Y = tf.get_mini_batch(self.batch_size, self.sequence_length)
            l = self.model.train_on_batch(X,Y)
            logger.log_scalar('loss', l[0], batches)
            logger.log_scalar('acc', l[1], batches)
            logger.log_scalar('epochs', tf.get_epochs(), batches)
            batches+=1
            if batches%50==0:
                if tf.get_epochs() - epochs > 0:
                    self.save_models()
                    epochs = tf.get_epochs()
                samp = self.sample(tf,512)
                print('Epoch {}'.format(tf.get_epochs()))
                print(samp)

    def sample_softmax(self, a, temperature=1.0):
        a = np.log(a) / temperature
        a = np.exp(a) / np.sum(np.exp(a))
        return np.random.choice(a.shape[0], p=a)

    def sample(self, tf, length, out_fp=None):
        sampled_hots = []
        self.pred_model.set_weights(self.model.get_weights())
        self.pred_model.reset_states()
        rows = np.random.randint(0,tf.vocab_size-1,size=(1,1))
        b = Progbar(length)
        print('Generating Text')
        for i in np.arange(0, length):
            next_row_index = self.sample_softmax(np.squeeze(self.pred_model.predict(rows), 0)[0], self.temp)
            sampled_hots.append(next_row_index)
            rows = np.array([next_row_index]).reshape((1,1))
            b.update(i)
        sampled_hots = np.array(sampled_hots)
        s_string = tf.indexes_to_string(sampled_hots)
        if out_fp is None:
            out_fp = 'samples/{}.txt'.format(self.name)
        with open(out_fp, 'w') as fp:
            fp.write(s_string)
        return s_string

    def save_models(self):
        self.model.save_weights('models/{}_weights.h5'.format(self.name))

    def load_models(self):
        self.model.load_weights('models/{}_weights.h5'.format(self.name))
        print('Loaded weights for prefix {}'.format(self.name))


if __name__ == '__main__':
    t_file = TextFile('trainTexts/tifu.txt')
    n = os.path.basename(t_file.fp).replace('.txt','')
    rnn = CharRNN(n,t_file.vocab_size,512,128,0.9)
    logger = Logger()
    print(rnn.model.summary())
    #rnn.load_models()
    #rnn.sample(t_file,4096)
    rnn.train_lstm(t_file,logger)

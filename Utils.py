from keras import Model
from keras.utils.generic_utils import Progbar
from keras.models import Sequential
from keras.callbacks import Callback
from keras.optimizers import Adam
from keras.layers import Dense, InputLayer, Activation, Input, Concatenate, Reshape
from keras.layers import GRU, TimeDistributed, Bidirectional, ConvLSTM2D
from io import BytesIO
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import os
import shutil

class ResetStateCallback(Callback):

    def __init__(self, logger, rnn, db):
        super(ResetStateCallback, self).__init__()
        self.logger = logger
        self.batches = 0
        self.rnn = rnn
        self.db = db
        self.bar = Progbar(2000000)
        self.songs_learned = 0

    def on_epoch_end(self, epoch, logs=None):
        for k,v in logs.items():
            self.logger.log_scalar(k, v, self.batches)
        self.rnn.temp = 0.9
        self.rnn.save_models()
        self.model.reset_states()


    def on_batch_end(self, batch, logs=None):
        for k,v in logs.items():
            self.logger.log_scalar(k, v, self.batches)
        self.batches+=1

class Logger(object):
    """Logging in tensorboard without tensorflow ops."""

    def __init__(self, log_dir='logs'):
        """Creates a summary writer logging to log_dir."""
        if os.path.exists(log_dir):
            shutil.rmtree(log_dir)
            print('removed log dir')
        os.makedirs(log_dir)
        self.writer = tf.summary.FileWriter(log_dir)
        self.writer_1 = tf.summary.FileWriter(log_dir+'/scalar1')
        self.writer_2 = tf.summary.FileWriter(log_dir+'/scalar2')

    def log_scalar(self, tag, value, step):
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag,
                                                     simple_value=value)])
        self.writer.add_summary(summary, step)
        self.writer.flush()

    def log_scalars(self, tag, value_1, value_2, step):
        """
            usefull for comparing values
        :param tag: name
        :param value_1: val 1
        :param value_2: val 2
        :param step: curent step
        :return: nothing dogggg
        """
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag,
                                                     simple_value=value_1)])
        self.writer_1.add_summary(summary, step)
        self.writer_1.flush()
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag,
                                                     simple_value=value_2)])
        self.writer_2.add_summary(summary, step)
        self.writer_2.flush()

    def log_image(self, tag, image, step):
        im_summaries = []
        np_data = np.squeeze(image)
        rescaled = (255.0 / np_data.max() * (np_data - np_data.min()))
        im = Image.fromarray(rescaled.astype(np.uint8))
        s = BytesIO()
        plt.imsave(s, im, format='png')

        img_sum = tf.Summary.Image(encoded_image_string=s.getvalue(), height=image.shape[0], width=image.shape[1])
        im_summaries.append(tf.Summary.Value(tag='%s/%d' % (tag, step), image=img_sum))

        summary = tf.Summary(value=im_summaries)
        self.writer.add_summary(summary, step)
        self.writer.flush()
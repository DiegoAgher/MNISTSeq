import numpy as np
import pandas as pd
import cv2

from keras.datasets import mnist
from keras.utils.np_utils import to_categorical


class MNISTSeqGen(object):
    def __init__(self, seq_length):
        self.seq_length = seq_length
        (self.x_train, self.y_train), (self.x_val, self.y_val) = mnist.load_data()

    def train_generate(self, batch_size, should_blur=False, probability=0.5):

        while True:

            batch_X, batch_y = [], []

            for i in range(batch_size):
                image_seq, seq_label = self.generate_mnist_seq(should_blur, probability)

                batch_X.append(image_seq)
                batch_y.append(seq_label)

            batch_X = np.array(batch_X)
            batch_y = np.array(batch_y)

            yield (batch_X, batch_y)

    def val_generate(self, batch_size, should_blur=False):

        while True:

            batch_X, batch_y = [], []

            for i in range(batch_size):
                image_seq, seq_label = self.generate_mnist_seq(should_blur=should_blur,
                                                               train_or_val='val')

                batch_X.append(image_seq)
                batch_y.append(seq_label)

            batch_X = np.array(batch_X)
            batch_y = np.array(batch_y)

            yield (batch_X, batch_y)

    def generate_mnist_seq(self, train_or_val='train', should_blur=False,
                           probability=0.5):
        dataset, labels = self.x_train, self.y_train
        if train_or_val != 'train':
            self.x_val, self.y_val

        random_chars = np.random.randint(0, len(dataset) - 1, self.seq_length)
        image_seq = []

        if not should_blur:
            image_seq = [(dataset[j]) for j in random_chars]
        else:
            for j in random_chars:
                if np.random.random(1) >= probability:
                    image_seq.append(cv2.medianBlur(dataset[j], 5))
                else:
                    image_seq.append(dataset[j])

        image_seq = np.concatenate(image_seq, axis=1)
        image_seq = image_seq.reshape(image_seq.shape[0], image_seq.shape[1], 1)

        seq_label = np.array([labels[j] for j in random_chars])
        seq_label = np.append([9], seq_label)
        seq_label = to_categorical(seq_label)
        seq_label = seq_label[1:]

        return image_seq, seq_label
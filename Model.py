from keras.optimizers import Adadelta, Adam
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Input, Dense, Activation
from keras.layers import Reshape, Lambda, BatchNormalization, Dropout
from keras.layers.merge import add, concatenate
from keras.models import Model
from keras.layers.recurrent import LSTM


def build_Model(num_classes, seq_lenght, input_shape=(28, 252, 1)):

    inputs = Input(name='x', shape=input_shape, dtype='float32')

    conv1 = Conv2D(8, (3, 3), padding='same', name='conv1',
                   kernel_initializer='he_normal')(inputs)
    conv1 = BatchNormalization()(conv1)
    conv1 = Activation('relu')(conv1)
    conv1 = MaxPooling2D(pool_size=(2, 2), name='max1')(conv1)

    conv2 = Conv2D(16, (3, 3), padding='same', name='conv2',
                   kernel_initializer='he_normal')(conv1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Activation('relu')(conv2)
    conv2 = MaxPooling2D(pool_size=(2, 2), name='max2')(conv2)

    conv3 = Conv2D(32, (3, 3), padding='same', name='conv3',
                   kernel_initializer='he_normal')(conv2)
    conv3 = BatchNormalization()(conv3)
    conv3 = Activation('relu')(conv3)
    conv3 = Conv2D(32, (3, 3), padding='same', name='conv4',
                   kernel_initializer='he_normal')(conv3)
    conv3 = BatchNormalization()(conv3)
    conv3 = Activation('relu')(conv3)
    conv3 = MaxPooling2D(pool_size=(1, 2), name='max3')(conv3)

    conv4 = Conv2D(32, (3, 3), padding='same', name='conv5',
                   kernel_initializer='he_normal')(conv3)
    conv4 = BatchNormalization()(conv4)
    conv4 = Activation('relu')(conv4)
    conv4 = Conv2D(32, (3, 3), padding='same', name='conv6')(conv4)
    conv4 = BatchNormalization()(conv4)
    conv4 = Activation('relu')(conv4)
    conv4 = MaxPooling2D(pool_size=(1, 2), name='max4')(conv4)

    conv5 = Conv2D(seq_lenght, (2, 2), padding='same', kernel_initializer='he_normal',
                   name='con7')(conv4)
    conv5 = BatchNormalization()(conv5)
    conv5 = Activation('relu')(conv5)
    dims = conv5.get_shape()
    reshape = Reshape(target_shape=(seq_lenght, int(dims[1] * dims[2])),
                      name='reshape')(conv5)
    reshape = Dense(128, activation='relu', kernel_initializer='he_normal',
                    name='dense2')(reshape)
    reshape = Dropout(0.5)(reshape)

    lstm_1 = LSTM(32, return_sequences=True, kernel_initializer='he_normal',
                  name='lstm1')(reshape)
    lstm_1 = BatchNormalization()(lstm_1)
    lstm_2 = LSTM(32, return_sequences=True, kernel_initializer='he_normal',
                  name='lstm2')(lstm_1)
    lstm_2 = BatchNormalization()(lstm_2)

    y_pred = Dense(num_classes, activation='softmax',
                   kernel_initializer='he_normal', name='output')(lstm_2)

    model = Model(inputs=inputs, outputs=y_pred)
    model.compile(Adam(lr=0.001), 'categorical_crossentropy', metrics=['accuracy'])

    return model

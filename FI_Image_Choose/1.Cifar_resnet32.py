#from https://blog.csdn.net/bryant_meng/article/details/81187434

import keras
import numpy as np
from keras.datasets import cifar10, cifar100
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
from keras.layers import Conv2D, Dense, Input, add, Activation, GlobalAveragePooling2D
from keras.callbacks import LearningRateScheduler, TensorBoard
from keras.models import Model
from keras import optimizers, regularizers
from keras import backend as K
#import tensorflow as tf
import os

stack_n            = 5
layers             = 6 * stack_n + 2
num_classes        = 10
batch_size         = 128
epochs             = 200
iterations         = 50000 // batch_size + 1
weight_decay       = 1e-4

log_filepath = './my_resnet_32_cifar/'
if not os.path.exists(log_filepath):
    os.mkdir(log_filepath)

def scheduler(epoch):
    if epoch < 80:
        return 0.1
    if epoch < 160:
        return 0.01
    return 0.001

# data
x_train = np.load('Cifar_set/Cifar_train_image.npy').reshape(-1,32,32,3)    # (50000, 32 32, 3)
y_train = np.load('Cifar_set/Cifar_train_label.npy')                        # (50000, 10)
print(x_train.shape)
print(x_train.shape[1:])

#x_test = np.load('Cifar_set/Cifar_test_image.npy').reshape(-1,32,32,3)   # (10000, 32 32, 3)
#y_test = np.load('Cifar_set/Cifar_test_label.npy')


def res_32(img_input):
    # input: 32x32x3 output: 32x32x16
    x = Conv2D(16, (3, 3), strides=(1, 1), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay),
               kernel_initializer="he_normal")(img_input)

    # res_block1 to res_block5 input: 32x32x16 output: 32x32x16
    for _ in range(5):
        b0 = BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
        a0 = Activation('relu')(b0)
        conv_1 = Conv2D(16, kernel_size=(3, 3), strides=(1, 1), padding='same',
                        kernel_regularizer=regularizers.l2(weight_decay),
                        kernel_initializer="he_normal")(a0)
        b1 = BatchNormalization(momentum=0.9, epsilon=1e-5)(conv_1)
        a1 = Activation('relu')(b1)
        conv_2 = Conv2D(16, kernel_size=(3, 3), strides=(1, 1), padding='same',
                        kernel_regularizer=regularizers.l2(weight_decay),
                        kernel_initializer="he_normal")(a1)

        x = add([x, conv_2])

    # res_block6 input: 32x32x16 output: 16x16x32
    b0 = BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
    a0 = Activation('relu')(b0)
    conv_1 = Conv2D(32, kernel_size=(3, 3), strides=(2, 2), padding='same',
                    kernel_regularizer=regularizers.l2(weight_decay),
                    kernel_initializer="he_normal")(a0)
    b1 = BatchNormalization(momentum=0.9, epsilon=1e-5)(conv_1)
    a1 = Activation('relu')(b1)
    conv_2 = Conv2D(32, kernel_size=(3, 3), strides=(1, 1), padding='same',
                    kernel_regularizer=regularizers.l2(weight_decay),
                    kernel_initializer="he_normal")(a1)

    projection = Conv2D(32, kernel_size=(1, 1), strides=(2, 2), padding='same',
                        kernel_regularizer=regularizers.l2(weight_decay),
                        kernel_initializer="he_normal")(a0)
    x = add([projection, conv_2])

    # res_block7 to res_block10 input: 16x16x32 output: 16x16x32
    for _ in range(1, 5):
        b0 = BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
        a0 = Activation('relu')(b0)
        conv_1 = Conv2D(32, kernel_size=(3, 3), strides=(1, 1), padding='same',
                        kernel_regularizer=regularizers.l2(weight_decay),
                        kernel_initializer="he_normal")(a0)
        b1 = BatchNormalization(momentum=0.9, epsilon=1e-5)(conv_1)
        a1 = Activation('relu')(b1)
        conv_2 = Conv2D(32, kernel_size=(3, 3), strides=(1, 1), padding='same',
                        kernel_regularizer=regularizers.l2(weight_decay),
                        kernel_initializer="he_normal")(a1)
        x = add([x, conv_2])

    # res_block11 input: 16x16x32 output: 8x8x64
    b0 = BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
    a0 = Activation('relu')(b0)
    conv_1 = Conv2D(64, kernel_size=(3, 3), strides=(2, 2), padding='same',
                    kernel_regularizer=regularizers.l2(weight_decay),
                    kernel_initializer="he_normal")(a0)
    b1 = BatchNormalization(momentum=0.9, epsilon=1e-5)(conv_1)
    a1 = Activation('relu')(b1)
    conv_2 = Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same',
                    kernel_regularizer=regularizers.l2(weight_decay),
                    kernel_initializer="he_normal")(a1)

    projection = Conv2D(64, kernel_size=(1, 1), strides=(2, 2), padding='same',
                        kernel_regularizer=regularizers.l2(weight_decay),
                        kernel_initializer="he_normal")(a0)
    x = add([projection, conv_2])

    # res_block12 to res_block15 input: 8x8x64 output: 8x8x64
    for _ in range(1, 5):
        b0 = BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
        a0 = Activation('relu')(b0)
        conv_1 = Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same',
                        kernel_regularizer=regularizers.l2(weight_decay),
                        kernel_initializer="he_normal")(a0)
        b1 = BatchNormalization(momentum=0.9, epsilon=1e-5)(conv_1)
        a1 = Activation('relu')(b1)
        conv_2 = Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same',
                        kernel_regularizer=regularizers.l2(weight_decay),
                        kernel_initializer="he_normal")(a1)
        x = add([x, conv_2])

    # Dense input: 8x8x64 output: 64
    x = BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
    x = Activation('relu')(x)
    x = GlobalAveragePooling2D()(x)

    # input: 64 output: 10
    x = Dense(10, activation='softmax', kernel_initializer="he_normal",
              kernel_regularizer=regularizers.l2(weight_decay))(x)

    return x

img_input = Input(shape=(32,32,3))
output = res_32(img_input)
resnet = Model(img_input, output)

# set optimizer
sgd = optimizers.SGD(lr=.1, momentum=0.9, nesterov=True)
resnet.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# set callback
tb_cb = TensorBoard(log_dir=log_filepath, histogram_freq=0)
change_lr = LearningRateScheduler(scheduler)
cbks = [change_lr,tb_cb]

# set data augmentation
datagen = ImageDataGenerator(horizontal_flip=True,
                             width_shift_range=0.125,
                             height_shift_range=0.125,
                             fill_mode='constant',cval=0.)

datagen.fit(x_train)

# start training
resnet.fit_generator(datagen.flow(x_train, y_train,batch_size=batch_size),
                     steps_per_epoch=iterations,
                     epochs=epochs,
                     callbacks=cbks,
                     verbose=1,
                     validation_data=(x_train, y_train))
resnet.save('my_resnet_32_cifar.h5')






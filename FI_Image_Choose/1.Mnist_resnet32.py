#from https://blog.csdn.net/bryant_meng/article/details/81187434

import keras
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
from keras.layers import Conv2D, Dense, Input, add, Activation, GlobalAveragePooling2D
from keras.callbacks import LearningRateScheduler, TensorBoard
from keras.models import Model
from keras import optimizers, regularizers
import os

stack_n            = 5
layers             = 6 * stack_n + 2
num_classes        = 10
batch_size         = 128
epochs             = 80
iterations         = 60000 // batch_size + 1
weight_decay       = 1e-4

log_filepath = './my_resnet_32_mnist/'
if not os.path.exists(log_filepath):
    os.mkdir(log_filepath)


def scheduler(epoch):
    if epoch < 20:
        return 0.1
    if epoch < 40:
        return 0.01
    return 0.001

# data
x_train = np.load('Mnist_set/Mnist_train_image.npy').reshape(-1,28,28,1)    # (60000, 28, 28, 1)
y_train = np.load('Mnist_set/Mnist_train_label.npy')                        # (60000, 10)
print(x_train.shape)
print(x_train.shape[1:])


def res_32(img_input):
    # input: 28x28x1 output: 28x28x16
    x = Conv2D(16, (3, 3), strides=(1, 1), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay),
               kernel_initializer="he_normal")(img_input)

    # res_block1 to res_block5 input: 28x28x16 output: 28x28x16
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

    # res_block6 input: 28x28x16 output: 14x14x32
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

    # res_block7 to res_block10 input: 14x14x32 output: 14x14x32
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

    # res_block11 input: 14x14x32 output: 7x7x64
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

    # res_block12 to res_block15 input: 7x7x64 output: 7x7x64
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

    # Dense input: 7x7x64 output: 64
    x = BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
    x = Activation('relu')(x)
    x = GlobalAveragePooling2D()(x)

    # input: 64 output: 10
    x = Dense(10, activation='softmax', kernel_initializer="he_normal",
              kernel_regularizer=regularizers.l2(weight_decay))(x)

    return x


img_input = Input(shape=(28,28,1))
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
datagen = ImageDataGenerator(width_shift_range=0.125,
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
resnet.save('my_resnet_32_mnist.h5')





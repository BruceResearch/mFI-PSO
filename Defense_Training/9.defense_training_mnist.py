

import tensorflow as tf
from keras.models import *
import keras
import numpy as np
import random
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
from keras.layers import Conv2D, Dense, Input, add, Activation, GlobalAveragePooling2D,Dropout
from keras.callbacks import LearningRateScheduler, TensorBoard
from keras.models import Model
from keras import optimizers, regularizers
from keras import backend as K
import os


stack_n            = 5
layers             = 6 * stack_n + 2
num_classes        = 10
batch_size         = 128
epochs             = 30
iterations         = 61220 // batch_size + 1
#iterations         = 1220 // batch_size + 1
weight_decay       = 1e-4

log_filepath = '\my_resnet_32_mnist_mix_adv'

def scheduler(epoch):
    if epoch < 10:
        return 0.1
    if epoch < 20:
        return 0.01
    return 0.001


# load data
x_train = np.load('/your_path_to_main_dir/FI_Image_Choose/adv_info/adv_Mnist/combine_train/x_mix_train.npy').reshape(-1,28,28,1) 
y_train = np.load('/your_path_to_main_dir/FI_Image_Choose/adv_info/adv_Mnist/combine_train/y_mix_train.npy')                     

x_adv_train = np.load('/your_path_to_main_dir/FI_Image_Choose/adv_info/adv_Mnist/combine_train/subset_of_success_x.npy').reshape(-1,28,28,1) 
y_adv_train = np.load('/your_path_to_main_dir/FI_Image_Choose/adv_info/adv_Mnist/combine_train/subset_of_success_y.npy')

x_test = np.load('/your_path_to_main_dir/FI_Image_Choose/adv_info/adv_Mnist/combine_test/x_mix_test.npy').reshape(-1,28,28,1) 
y_test = np.load('/your_path_to_main_dir/FI_Image_Choose/adv_info/adv_Mnist/combine_test/y_mix_test.npy')

x_adv_test = np.load('/your_path_to_main_dir/FI_Image_Choose/adv_info/adv_Mnist/combine_test/Mnist_test_adv_x.npy').reshape(-1,28,28,1)
y_adv_test = np.load('/your_path_to_main_dir/FI_Image_Choose/adv_info/adv_Mnist/combine_test/Mnist_test_adv_y.npy')

x_original_test = np.load('/your_path_to_main_dir/FI_Image_Choose/Mnist_set/Mnist_test_image.npy').reshape(-1,28,28,1) 
y_original_test = np.load('/your_path_to_main_dir/FI_Image_Choose/Mnist_set/Mnist_test_label.npy')    


def res_32(img_input):
    # input: 28x28x1 output: 28x28x16
    x = Conv2D(16, (3, 3), strides=(1, 1), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay),
               kernel_initializer="he_normal")(img_input)

    # res_block1 to res_block5 input: 28x28x16 output: 28x28x16
    for _ in range(5):
        b0 = BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
        a0 = Activation('relu')(b0)
        #a0 = Dropout(0.5)(a0)
        conv_1 = Conv2D(16, kernel_size=(3, 3), strides=(1, 1), padding='same',
                        kernel_regularizer=regularizers.l2(weight_decay),
                        kernel_initializer="he_normal")(a0)
        b1 = BatchNormalization(momentum=0.9, epsilon=1e-5)(conv_1)
        a1 = Activation('relu')(b1)
        #a1 = Dropout(0.5)(a1)
        conv_2 = Conv2D(16, kernel_size=(3, 3), strides=(1, 1), padding='same',
                        kernel_regularizer=regularizers.l2(weight_decay),
                        kernel_initializer="he_normal")(a1)

        x = add([x, conv_2])

    # res_block6 input: 28x28x16 output: 14x14x32
    b0 = BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
    a0 = Activation('relu')(b0)
    #a0 = Dropout(0.5)(a0)
    conv_1 = Conv2D(32, kernel_size=(3, 3), strides=(2, 2), padding='same',
                    kernel_regularizer=regularizers.l2(weight_decay),
                    kernel_initializer="he_normal")(a0)
    b1 = BatchNormalization(momentum=0.9, epsilon=1e-5)(conv_1)
    a1 = Activation('relu')(b1)
    #a1 = Dropout(0.5)(a1)
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
        #a0 = Dropout(0.5)(a0)
        conv_1 = Conv2D(32, kernel_size=(3, 3), strides=(1, 1), padding='same',
                        kernel_regularizer=regularizers.l2(weight_decay),
                        kernel_initializer="he_normal")(a0)
        b1 = BatchNormalization(momentum=0.9, epsilon=1e-5)(conv_1)
        a1 = Activation('relu')(b1)
        #a1 = Dropout(0.5)(a1)
        conv_2 = Conv2D(32, kernel_size=(3, 3), strides=(1, 1), padding='same',
                        kernel_regularizer=regularizers.l2(weight_decay),
                        kernel_initializer="he_normal")(a1)
        x = add([x, conv_2])

    # res_block11 input: 14x14x32 output: 7x7x64
    b0 = BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
    a0 = Activation('relu')(b0)
    #a0 = Dropout(0.5)(a0)
    conv_1 = Conv2D(64, kernel_size=(3, 3), strides=(2, 2), padding='same',
                    kernel_regularizer=regularizers.l2(weight_decay),
                    kernel_initializer="he_normal")(a0)
    b1 = BatchNormalization(momentum=0.9, epsilon=1e-5)(conv_1)
    a1 = Activation('relu')(b1)
    #a1 = Dropout(0.5)(a1)
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
        #a0 = Dropout(0.5)(a0)
        conv_1 = Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same',
                        kernel_regularizer=regularizers.l2(weight_decay),
                        kernel_initializer="he_normal")(a0)
        b1 = BatchNormalization(momentum=0.9, epsilon=1e-5)(conv_1)
        a1 = Activation('relu')(b1)
        #a1 = Dropout(0.5)(a1)
        conv_2 = Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same',
                        kernel_regularizer=regularizers.l2(weight_decay),
                        kernel_initializer="he_normal")(a1)
        x = add([x, conv_2])

    # Dense input: 7x7x64 output: 64
    x = BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
    x = Activation('relu')(x)
    #X = Dropout(0.2)(x)
    x = GlobalAveragePooling2D()(x)

    # input: 64 output: 10
    x = Dense(10, activation='softmax', kernel_initializer="he_normal",
              kernel_regularizer=regularizers.l2(weight_decay))(x)

    return x


# load model
resnet = load_model('/your_path_to_main_dir/FI_Image_Choose/my_resnet_32_mnist.h5')

# set optimizer
sgd = optimizers.SGD(lr=.001, momentum=0.9, nesterov=True)
resnet.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

#get a prelimiary testing
results4 = resnet.evaluate(x_test, y_test, batch_size=128)
print("ours adv trained test on our test mix loss, test acc:", results4)

results5 = resnet.evaluate(x_adv_test, y_adv_test, batch_size=128)
print("ours adv trained test on our adv test only loss, test acc:", results5)

results6 = resnet.evaluate(x_original_test, y_original_test, batch_size=128)
print("ours adv trained test on original test loss, test acc:", results6)

results7 = resnet.evaluate(x_train, y_train, batch_size=128)
print("ours adv trained test on mix training loss, test acc:", results7)

results9 = resnet.evaluate(x_adv_train, y_adv_train, batch_size=128)
print("ours adv trained test on adv training only loss, test acc:", results9)

# set callback
tb_cb = TensorBoard(log_dir=log_filepath, histogram_freq=0)
change_lr = LearningRateScheduler(scheduler)
cbks = [change_lr,tb_cb]


# set data augmentation
datagen = ImageDataGenerator(horizontal_flip=False,
                             width_shift_range=0.125,
                             height_shift_range=0.125,
                             preprocessing_function=add_noise,
                             fill_mode='constant',cval=0.)

datagen.fit(x_train)


# start training
resnet.fit(x_train, y_train,
                     steps_per_epoch=iterations,
                     epochs=epochs,
                     callbacks=cbks,
                     verbose=1,
                     validation_data=(x_train, y_train),
                     validation_batch_size = 128,
                     validation_freq =1)

resnet.save('30e_adv_trained_mnist_resenet32.h5')


#test the results 
results1 = resnet.evaluate(x_test, y_test, batch_size=128)
print("ours adv trained test on our test mix loss, test acc:", results1)

results2 = resnet.evaluate(x_adv_test, y_adv_test, batch_size=128)
print("ours adv trained test on our test loss, test acc:", results2)

results3 = resnet.evaluate(x_original_test, y_original_test, batch_size=128)
print("ours adv trained test on original test loss, test acc:", results3)

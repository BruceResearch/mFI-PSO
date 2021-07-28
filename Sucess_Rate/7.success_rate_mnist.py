from keras.models import *
import keras
import numpy as np
import glob
from keras.utils import to_categorical
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
epochs             = 50
iterations         = 60000 // batch_size + 1
weight_decay       = 1e-4



def scheduler(epoch):
    if epoch < 10:
        return 0.05
    if epoch < 20:
        return 0.01
    return 0.001

# data
x_train = np.load('/your_path_to_main_dir/FI_Image_Choose/adv_info/adv_Mnist/combine_train/Mnist_train_adv_x.npy').reshape(-1,28,28,1)   
y_train = np.load('/your_path_to_main_dir/FI_Image_Choose/adv_info/adv_Mnist/combine_train/Mnist_train_adv_y.npy').reshape(-1,10)                   

print(x_train.shape)
print(y_train.shape)

x_test = np.load('/your_path_to_main_dir/FI_Image_Choose/adv_info/adv_Mnist/combine_test/Mnist_test_adv_x.npy').reshape(-1,28,28,1)   
y_test = np.load('/your_path_to_main_dir/FI_Image_Choose/adv_info/adv_Mnist/combine_test/Mnist_test_adv_y.npy').reshape(-1,10)                     

print(x_test.shape)
print(y_test.shape)


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

# load model
resnet = load_model('/your_path_to_main_dir/FI_Image_Choose/my_resnet_32_mnist.h5')

#test for success rate 
train_result = resnet.evaluate(x_train,y_train, batch_size=128)
print("original trained resenet 32 tested on adversarial train image loss, test acc:", train_result)

test_result = resnet.evaluate(x_test,y_test, batch_size=128)
print("original trained resenet 32 tested on oadversarial test image loss, test acc:", test_result)

#filter the success adversarial images only for defense training 
pred = resnet.predict(x_train)
indices = [i for i,v in enumerate(pred) if np.argmax(pred[i])!=np.argmax(y_train[i])]
subset_of_success_x = [x_train[i].reshape(1,28,28,1) for i in indices ]
subset_of_success_y = [y_train[i].reshape(1,10) for i in indices ]
subset_of_success_x = np.concatenate(subset_of_success_x, axis=0)
subset_of_success_y = np.concatenate(subset_of_success_y, axis=0)
print((subset_of_success_x).shape)
print((subset_of_success_y).shape)
np.save('/your_path_to_main_dir/FI_Image_Choose/adv_info/adv_Mnist/combine_train/subset_of_success_x.npy',subset_of_success_x)
np.save('/your_path_to_main_dir/FI_Image_Choose/adv_info/adv_Mnist/combine_train/subset_of_success_y.npy',subset_of_success_y)

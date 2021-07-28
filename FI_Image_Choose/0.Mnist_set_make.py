
import numpy as np
import input_data
import random


mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

train_images_org = mnist.train.images  # (55000, 784)
train_labels_org = mnist.train.labels  # (55000, 10)

validation_images_org = mnist.validation.images  # (5000, 784)
validation_labels_org = mnist.validation.labels  # (5000, 10)

test_images = mnist.test.images  # (10000, 784)
test_labels = mnist.test.labels  # (10000, 10)

train_images_all = np.vstack((train_images_org,validation_images_org))
train_labels_all = np.vstack((train_labels_org,validation_labels_org))


np.save('./Mnist_set/Mnist_train_image.npy',train_images_all)   # (60000,784)
np.save('./Mnist_set/Mnist_train_label.npy',train_labels_all)   # (60000,10)
np.save('./Mnist_set/Mnist_test_image.npy',test_images)       # (10000,784)
np.save('./Mnist_set/Mnist_test_label.npy',test_labels)       # (10000,784)






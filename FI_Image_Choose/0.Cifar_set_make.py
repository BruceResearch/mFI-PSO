
from keras.datasets import cifar10
import numpy as np
from keras.utils import to_categorical

(x_img_train,y_label_train),(x_img_test, y_label_test)=cifar10.load_data()
print('train_image :',x_img_train.shape)      #(50000, 32, 32, 3)
print('train_label :',y_label_train.shape)    #(50000, 1)
print('test_image :',x_img_test.shape)        #(10000, 32, 32, 3)
print('test_label :',y_label_test.shape)      #(10000, 1)

#  onehot
y_label_train_10 = np.zeros((len(x_img_train),10))
for i in range(len(x_img_train)):
    y_label_train_10[i] = to_categorical(y_label_train[i][0],num_classes=10)
print('train_label_10 :',y_label_train_10.shape)    #(50000, 10)

#  onehot
y_label_test_10 = np.zeros((len(x_img_test),10))
for i in range(len(x_img_test)):
    y_label_test_10[i] = to_categorical(y_label_test[i][0],num_classes=10)
print('test_label_10 :',y_label_test_10.shape)    #(10000, 10)


np.save('./Cifar_set/Cifar_train_image.npy',x_img_train)        # (50000, 32，32，3）
np.save('./Cifar_set/Cifar_train_label.npy',y_label_train_10)   # (50000, 10)
np.save('./Cifar_set/Cifar_test_image.npy',x_img_test)          # (10000, 32, 32, 3)
np.save('./Cifar_set/Cifar_test_label.npy',y_label_test_10)     # (10000, 10)



import numpy as np
import numpy
import glob
import os
from natsort import natsorted, ns

#make x adv train
x_train_files = natsorted(glob.glob(r'/your_path_to_main_dir/FI_Image_Choose/adv_info/adv_Mnist/train/x' + '/*.npy'))
x_img_train = np.load(x_train_files[0]).reshape(1,28,28,1)
print(x_img_train.shape)
for f in x_train_files [1:]:
    x= np.load(f)
    x_img_train = np.concatenate((x_img_train,x.reshape(1,28,28,1)),axis=0)
    

print('train_image :',x_img_train.shape)   
np.save('/your_path_to_main_dir/FI_Image_Choose/adv_info/adv_Mnist/combine_train/Mnist_train_adv_x.npy',x_img_train)

#make y adv train
y_train_files = natsorted(glob.glob(r'/your_path_to_main_dir/FI_Image_Choose/adv_info/adv_Mnist/train/y' + '/*.npy'))
train_target_adv = np.load(y_train_files[0]).reshape(1,10)
print(train_target_adv.shape)
for f in y_train_files [1:]:
    y = np.load(f)
    train_target_adv = np.concatenate((train_target_adv,y.reshape(1,10)),axis=0)

print('train_target :',train_target_adv.shape)   
np.save('/your_path_to_main_dir/FI_Image_Choose/adv_info/adv_Mnist/combine_train/Mnist_train_adv_y.npy',train_target_adv)

#make x adv test
test_files = natsorted(glob.glob(r'/your_path_to_main_dir/FI_Image_Choose/adv_info/adv_Mnist/test/x' + '/*.npy'))
x_img_test = np.load(test_files[0]).reshape(1,28,28,1)
for f in test_files [1:]:
    x= np.load(f)
    x_img_test = np.concatenate((x_img_test,x.reshape(1,28,28,1)),axis=0)

print('test_image :',x_img_test.shape)  
np.save('/your_path_to_main_dir/FI_Image_Choose/adv_info/adv_Mnist/combine_test/Mnist_test_adv_x.npy',x_img_test)

#make y adv test
y_test_files = natsorted(glob.glob(r'/your_path_to_main_dir/FI_Image_Choose/adv_info/adv_Mnist/test/y' + '/*.npy'))
test_target_adv = np.load(y_test_files[0]).reshape(1,10)
for f in y_test_files [1:]:
    y = np.load(f)
    test_target_adv = np.concatenate((test_target_adv,y.reshape(1,10)),axis=0)

print('train_target :',test_target_adv.shape)   
np.save('/your_path_to_main_dir/FI_Image_Choose/adv_info/adv_Mnist/combine_test/Mnist_test_adv_y.npy',test_target_adv)

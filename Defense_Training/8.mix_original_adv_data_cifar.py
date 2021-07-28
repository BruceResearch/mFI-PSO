import numpy as np
import numpy
import glob
import os

x_adv_train = np.load('/your_path_to_main_dir/FI_Image_Choose/adv_info/adv_Cifar/combine_train/subset_of_success_x.npy')
y_adv_train = np.load('/your_path_to_main_dir/FI_Image_Choose/adv_Cifar/fix4/combine_train/subset_of_success_y.npy')

x_original_train = np.load('/your_path_to_main_dir/FI_Image_Choose/Cifar_set/Cifar_train_image.npy').reshape(-1,32,32,3)
y_original_train = np.load('/your_path_to_main_dir/FI_Image_Choose/Cifar_set/Cifar_train_label.npy')    

x_mix_train = np.concatenate((x_adv_train, x_original_train), axis=0)
y_mix_train = np.concatenate((y_adv_train, y_original_train), axis=0)


print(x_mix_train.shape)
print(y_mix_train.shape)

np.save('/your_path_to_main_dir/FI_Image_Choose/adv_info/adv_Cifar/combine_train/x_mix_train.npy',x_mix_train)
np.save('/your_path_to_main_dir/FI_Image_Choose/adv_info/adv_Cifar/combine_train/y_mix_train.npy',y_mix_train)

x_adv_test = np.load('/your_path_to_main_dir/FI_Image_Choose/adv_info/adv_Cifar/combine_test/cifar_test_adv_x.npy')
y_adv_test= np.load('/your_path_to_main_dir/FI_Image_Choose/adv_info/adv_Cifar/combine_test/cifar_test_adv_y.npy')

x_original_test = np.load('/your_path_to_main_dir/FI_Image_Choose/Cifar_set/Cifar_test_image.npy').reshape(-1,32,32,3)
y_original_test = np.load('/your_path_to_main_dir/FI_Image_Choose/Cifar_set/Cifar_test_label.npy')    

x_mix_test = np.concatenate((x_adv_test, x_original_test), axis=0)
y_mix_test = np.concatenate((y_adv_test, y_original_test), axis=0)

print(x_mix_test.shape)
print(y_mix_test.shape)

np.save('/your_path_to_main_dir/FI_Image_Choose/adv_info/adv_Cifar/combine_test/x_mix_test.npy',x_mix_test)
np.save('/your_path_to_main_dir/FI_Image_Choose/adv_info/adv_Cifar/combine_test/y_mix_test.npy',y_mix_test)

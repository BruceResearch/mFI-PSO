import numpy as np
import numpy

x_fix_adv = np.load('/your_path_to_main_dir/FI_Image_Choose/adv_info/adv_Mnist/train/combine_train/Mnist_train_adv_x.npy')
#print(x_fix_adv[1])
x_original = np.load('/your_path_to_main_dir/FI_Image_Choose/Mnist_set/Mnist_train_image.npy').reshape(-1,28,28,1)

array_pic_num_cifar_all = np.load('/your_path_to_main_dir/FI_Image_Choose/adv_info/sample_info/Mnist_sheet(%d)_situation(1)_FI_below(0.20)_pro_y_target(0.00)_array.npy'%(2))
cut_start = 0
cut_end = len(array_pic_num_cifar_all)
train_adv=[]
for i_pic in range(cut_start,cut_end,1):
    pic_num= array_pic_num_cifar_all[i_pic][2]
    train_adv.append(x_original[int(pic_num)])

x_original_image = np.concatenate(train_adv).reshape(-1,28,28,1)
x_original_image = x_original_image[:1305] 
print(x_original_image.shape)
#print(x_original_image[1])
abs_difference = []
for i in range(1305):

	abs_difference.append(np.max(np.absolute(x_fix_adv[i] - x_original_image[i]), axis = (0,1)))



average = sum(abs_difference) / len(abs_difference)
print("mean:",average)
std = numpy.std(abs_difference)
print("standard deviation:", std)
max_v = max(abs_difference)
print("max:", max_v)
quantile = np.percentile(abs_difference, 95)
print("95% quantile:",quantile)
import numpy as np
import Mnist_pixel_FI
import Mnist_PSO_critical_point_adv_make



# training set：2 / testing set：3
sheet_id = 2

your_path = '/your_path_to_main_dir'
array_pic_num_mnist_all = np.load(str(your_path) + '/FI_Image_Choose/adv_info/sample_info/Mnist_sheet(%d)_situation(1)_FI_below(0.20)_pro_y_target(0.00)_array.npy'%(sheet_id))

cut_start = 0
cut_end =  len(array_pic_num_mnist_all)        # 136 / 26

for i_pic in range(cut_start,cut_end,1):  #  [0:correct_class, 1:target_class, 2:pic_num]

    pic_num_i = array_pic_num_mnist_all[i_pic][2]
    correct_class_i = array_pic_num_mnist_all[i_pic][0]
    target_class_i = array_pic_num_mnist_all[i_pic][1]

    # compute pixel FI
    FI_adv_pred_array = Mnist_pixel_FI.mnist_pixel_FI(pic_num=pic_num_i, target_class=target_class_i, sheet_id=sheet_id)  ## (1,784)

    flag = 0.01
    pixel_FI_array_flag = np.zeros((1,784))
    for i in range(784):
        if FI_adv_pred_array[0][i]>= flag:
            pixel_FI_array_flag[0][i] = 1
    m = np.count_nonzero(pixel_FI_array_flag.reshape(1, -1), axis=1)[0]   #

    # x_adv
    x_adv_of_i, y_adv_of_i  = Mnist_PSO_critical_point_adv_make.PSO_critical_point(pic_num=pic_num_i,
                                                                      pixel_FI_array=FI_adv_pred_array, m=m,
                                                                      correct_class=correct_class_i,
                                                                      target_class=target_class_i,
                                                                      sheet_id=sheet_id)  # (1,3072)
    np.save(str(your_path) + '/FI_Image_Choose/adv_info/adv_Mnist/train/x/adv(%d)_mnist(%d)_sheet(%d)_situation(1)_FI_below(0.20)_pro_y_target(0.20)_POS_critical_point_adv_array.npy' % (i_pic,pic_num_i, sheet_id),x_adv_of_i)
    np.save(str(your_path) + '/FI_Image_Choose/adv_info/adv_Mnist/train/y/adv(%d)_mnist(%d)_sheet(%d)_situation(1)_FI_below(0.20)_pro_y_target(0.20)_POS_critical_point_adv_array.npy' % (i_pic,pic_num_i, sheet_id),y_adv_of_i)


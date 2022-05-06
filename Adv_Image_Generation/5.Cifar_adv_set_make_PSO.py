import numpy as np
import Cifar_pixel_FI
import Cifar_PSO_critical_point_adv_make


# training set：2 / testing set：3
sheet_id = 2

your_path = '/your_path_to_main_dir'
array_pic_num_cifar_all = np.load(str(your_path) + '/FI_Image_Choose/adv_info/sample_info/Cifar_sheet(%d)_situation(1)_FI_below(0.01)_pro_y_target(0.01)_array.npy'%(sheet_id))


cut_start = 0
cut_end = len(array_pic_num_cifar_all)  

x_adv_sum_list = []
for i_pic in range(cut_start,cut_end,1): 

    
    pic_num_i = array_pic_num_cifar_all[i_pic][2]
    correct_class_i = array_pic_num_cifar_all[i_pic][0]
    target_class_i = array_pic_num_cifar_all[i_pic][1]

    road_str_pixel_FI_i = str(your_path) + '/FI_Image_Choose/adv_info/adv_Cifar/train/adv(%d)_Cifar(%d)_sheet_id(%d)_pixel_FI.png' % (i_pic,pic_num_i, sheet_id)

    
    FI_adv_pred_array = Cifar_pixel_FI.cifar_pixel_FI(road_str = road_str_pixel_FI_i,pic_num=pic_num_i, target_class=target_class_i,
                                                      sheet_id=sheet_id)  ## (1,3072)


    flag = 0.01
    pixel_FI_array_flag = np.zeros((1,3072))
    for i in range(3072):
        if FI_adv_pred_array[0][i]>= flag:
            pixel_FI_array_flag[0][i] = 1
    m = np.count_nonzero(pixel_FI_array_flag.reshape(1, -1), axis=1)[0]  

    road_str_adv_i =  str(your_path) + '/FI_Image_Choose/adv_info/adv_Cifar/train/adv(%d)_Cifar(%d)_sheet(%d)_situation(1)_FI_below(0.01)_pro_y_target(0.10)_POS_critical_point_adv' %(i_pic,pic_num_i, sheet_id)

    # making adv
    x_adv_of_i, y_adv_of_i  = Cifar_PSO_critical_point_adv_make.PSO_critical_point(pic_num=pic_num_i,
                                                                      pixel_FI_array=FI_adv_pred_array, m=m,
                                                                      correct_class=correct_class_i,
                                                                      target_class=target_class_i,
                                                                      road_str_adv = road_str_adv_i,
                                                                      sheet_id=sheet_id)  # (1,3073) #(0: success_flag  , 1~3072: x_adv)
    np.save(str(your_path) + '/FI_Image_Choose/adv_info/adv_Cifar/train/x/adv(%d)_Cifar(%d)_sheet(%d)_situation(1)_FI_below(0.01)_pro_y_target(0.10)_POS_critical_point_adv_array.npy' % (i_pic,pic_num_i, sheet_id),x_adv_of_i)
    np.save(str(your_path) + '/FI_Image_Choose/adv_info/adv_Cifar/train/y/adv(%d)_Cifar(%d)_sheet(%d)_situation(1)_FI_below(0.01)_pro_y_target(0.10)_POS_critical_point_adv_array.npy' % (i_pic,pic_num_i, sheet_id),y_adv_of_i)



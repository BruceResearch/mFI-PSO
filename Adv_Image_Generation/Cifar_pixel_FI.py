

from keras.models import *
import numpy as np
import tensorflow as tf
import scipy as sp
import matplotlib.pyplot as plt
from tensorflow.keras import backend as K
tf.compat.v1.disable_eager_execution()

K.set_learning_phase(0)

# hyper parameter
epsilon = 1e-3
n_class = 10

def cifar_pixel_FI(road_str,pic_num,target_class,sheet_id = 2):
    # load images
    if sheet_id == 3:
        x_test = np.load('/your_path_to_main_dir/FI_Image_Choose/Cifar_set/Cifar_test_image.npy').reshape(-1, 32, 32, 3)  # (10000, 32 32, 3)
        y_test = np.load('/your_path_to_main_dir/FI_Image_Choose/Cifar_set/Cifar_test_label.npy')
        x_adv = x_test[int(pic_num)]    # x_adv (32, 32, 3)
        y_adv = y_test[int(pic_num)]
    else:
        x_train = np.load('/your_path_to_main_dir/FI_Image_Choose/Cifar_set/Cifar_train_image.npy').reshape(-1, 32, 32, 3)  # (50000, 32 32, 3)
        y_train = np.load('/your_path_to_main_dir/FI_Image_Choose/Cifar_set/Cifar_train_label.npy')  # (50000, 10)
        x_adv = x_train[int(pic_num)]   # x_adv (32, 32, 3)
        y_adv = y_train[int(pic_num)]

    # load model
    resnet32 = load_model('/your_path_to_main_dir/FI_Image_Choose/my_resnet_32_cifar.h5')

    # caulculating FI
    grad_K = tf.concat(axis=1, values=[tf.concat(axis=0, values=[K.flatten(b)[..., None] for b in K.gradients(resnet32.output[0, k], resnet32.input)]) for k in range(n_class)])
    iterate = K.function([resnet32.input], [grad_K, resnet32.output])

    grad, pred_P = iterate([x_adv[None]])  # (3072,10) ,(1,10)

    FI_adv_pred_array=np.zeros((1,3072))

    for pixel in range(3072):
        grad_fix = grad[pixel,:].reshape((1,-1))   #(1,10)  #grad[pixel,:].shape=(10,)

        L0 = grad_fix @ np.diag(((pred_P[0,]) ** 0.5 + epsilon) ** -1)

        f_grad_pred = ((grad[pixel, target_class] / (pred_P[0, target_class] + epsilon)).T).reshape(1,-1)

        B0, D_L0, A0 = sp.linalg.svd(L0, full_matrices=False)
        rank_L0 = sum(D_L0 > epsilon)

        if rank_L0 > 0:
            B0 = B0[:, :rank_L0]
            A0 = np.diag(D_L0[:rank_L0]) @ A0[:rank_L0, :]

            U_A, D_0, _ = sp.linalg.svd(A0 @ A0.T, full_matrices=True)
            D_0_inv = np.diag(D_0 ** -1)
            D_0_inv_sqrt = np.diag(D_0 ** -0.5)

            U_0 = B0 @ U_A

            nabla_f_pred = f_grad_pred @ U_0 @ D_0_inv_sqrt

            FI_adv_pred_array[0][pixel] = nabla_f_pred @ nabla_f_pred.T


   
    FI_test_pred_array = FI_adv_pred_array.reshape(32, 32, 3)

    FI_test_pred_red = FI_test_pred_array[:, :, 0].reshape((32, 32))  # red_channel_FI
    FI_test_pred_green = FI_test_pred_array[:, :, 1].reshape((32, 32))  # green_channel_FI
    FI_test_pred_blue = FI_test_pred_array[:, :, 2].reshape((32, 32))  # blue_channel_FI

    x_adv_red = x_adv[:, :, 0]  # red_channel_pixel
    x_adv_green = x_adv[:, :, 1]  # green_channel_pixel
    x_adv_blue = x_adv[:, :, 2]  # blue_channel_pixel


    # plot
    fig = plt.figure(figsize=(12,6))

    # original image
    ax1 = fig.add_subplot(1,4,1)
    ax1.imshow(x_adv.reshape((32,32,3)))   

    # model prediction distribution
    ax2 = fig.add_subplot(1, 4, 2)
    p = pred_P[0]  # p为img的网络输出概率
    correct_class = int(np.argmax(y_adv))
    topk_label = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    topk = list(np.argsort(p)[::-1])   
    topk_label_sort = [topk_label[i] for i in topk]
    topprobs = p[topk] 
    barlist = ax2.bar(range(10), topprobs)
    if target_class in topk:
        barlist[topk.index(target_class)].set_color('r')
    if correct_class in topk:
        barlist[topk.index(correct_class)].set_color('g')
    plt.sca(ax2)
    plt.ylim([0, 1.1])
    plt.xticks(range(10), [i for i in topk_label_sort], rotation='vertical')  # 对应各类类做柱状图的横坐标，对应概率从高到低排序
    fig.subplots_adjust(bottom=0.5)

    
    ax3 = fig.add_subplot(3, 4, 3)
    ax3.imshow(x_adv_red, cmap='Reds')

 
    ax5 = fig.add_subplot(3, 4, 7)
    ax5.imshow(x_adv_green, cmap='Greens')


    ax8 = fig.add_subplot(3, 4, 11)
    ax8.imshow(x_adv_blue, cmap='Blues')


    ax4 = fig.add_subplot(3, 4, 4)
    im4 = ax4.imshow(FI_test_pred_red, cmap='brg')
    plt.colorbar(im4, ticks=[0.0, 0.5, 1.0, 1.5, 2.0], shrink=0.7)

    ax6 = fig.add_subplot(3, 4, 8)
    im6 = ax6.imshow(FI_test_pred_green, cmap='brg')
    plt.colorbar(im6, ticks=[0.0, 0.5, 1.0, 1.5, 2.0], shrink=0.7)

    ax9 = fig.add_subplot(3, 4, 12)
    im9 = ax9.imshow(FI_test_pred_blue, cmap='brg')
    plt.colorbar(im9, ticks=[0.0, 0.5, 1.0, 1.5, 2.0], shrink=0.7)


    plt.savefig(road_str)
    plt.show()

    return FI_adv_pred_array # (1,3072)


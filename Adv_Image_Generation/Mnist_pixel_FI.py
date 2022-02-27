

from keras.models import *
import numpy as np
import tensorflow as tf
import scipy as sp
import matplotlib.pyplot as plt
from tensorflow.keras import backend as K

tf.compat.v1.disable_eager_execution()

plt.rcParams['font.sans-serif']=['SimSun']
plt.rcParams['axes.unicode_minus'] = False

K.set_learning_phase(0)

# 参数
epsilon = 1e-3       
n_class = 10         
your_path = '/your_path_to_main_dir'

def mnist_pixel_FI(road_str,pic_num,target_class,sheet_id = 2):
    # 加载选定图片
    if sheet_id == 3:
        x_test = np.load(str(your_path) + '/FI_Image_Choose/Mnist_set/Mnist_test_image.npy').reshape(-1, 28, 28, 1)  # (10000, 28, 28, 1)
        y_test = np.load(str(your_path) + '/FI_Image_Choose/Mnist_test_label.npy')  # (10000,10)
        x_adv = x_test[int(pic_num)]    # x_adv (32, 32, 3)
        y_adv = y_test[int(pic_num)]
    else:
        x_train = np.load(str(your_path) + '/FI_Image_Choose/Mnist_set/Mnist_train_image.npy').reshape(-1, 28, 28, 1)  # (60000, 28, 28, 1)
        y_train = np.load(str(your_path) + '/FI_Image_Choose/Mnist_set/Mnist_train_label.npy')  # (60000, 10)
        x_adv = x_train[int(pic_num)]   # x_adv (32, 32, 3)
        y_adv = y_train[int(pic_num)]


    # 载入模型
    resnet32 = load_model(str(your_path) + '/FI_Image_Choose/my_resnet_32_mnist.h5')

    # 计算pixel_FI（一通道）
    grad_K = tf.concat(axis=1, values=[tf.concat(axis=0, values=[K.flatten(b)[..., None] for b in K.gradients(resnet32.output[0, k], resnet32.input)]) for k in range(n_class)])
    iterate = K.function([resnet32.input], [grad_K, resnet32.output])

    # x_adv (28, 28)
    grad, pred_P = iterate([x_adv[None]])  # (784,10) ,(1,10)

    FI_adv_pred_array=np.zeros((1,784))

    for pixel in range(784):

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


    # 绘图
    fig = plt.figure(figsize=(10,5))

    # 原图
    ax1 = fig.add_subplot(1,3,1)
    ax1.imshow(x_adv.reshape((28, 28)),cmap='Greys_r')   #(M, N, 3): an image with RGB values (0-1 float or 0-255 int).

    # 模型预测概率分布
    ax2 = fig.add_subplot(1, 3, 2)
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
    plt.xticks(range(10), [i for i in topk_label_sort], rotation='vertical')  
    fig.subplots_adjust(bottom=0.5)

    # FI
    ax3 = fig.add_subplot(1,3,3)
    im3 = ax3.imshow(FI_adv_pred_array.reshape(28,28),cmap='brg')
    plt.colorbar(im3,  shrink=0.3)
    #plt.colorbar(im3,ticks=[0.0, 0.5, 1.0 ,1.5,2.0], shrink = 0.3)

    plt.savefig(road_str)
    plt.show()

    return FI_adv_pred_array # (1,784)

#mnist_pixel_FI(pic_num=10450,target_class=6,sheet_id = 2)



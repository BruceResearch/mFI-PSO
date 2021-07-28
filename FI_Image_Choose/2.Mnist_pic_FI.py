import numpy as np
import scipy as sp
import tensorflow as tf
from keras.models import *
import xlwt
from tensorflow.keras import backend as K

tf.compat.v1.disable_eager_execution()

K.set_learning_phase(0)

epsilon = 1e-3
n_classes  = 10

# 载入模型
resnet32 = load_model('my_resnet_32_mnist.h5')

# data
## train:
x_train = np.load('Mnist_set/Mnist_train_image.npy').reshape(-1,28,28,1)    # (60000, 28, 28, 1)
y_train = np.load('Mnist_set/Mnist_train_label.npy')                        # (60000, 10)
y_train_dig = np.argmax(y_train,axis=1)   # (60000,)

y_train_pred = resnet32.predict(x_train)  # (60000, 10)
y_train_pred_dig = np.argmax(y_train_pred,axis=1)  # (60000,)

## test_set
x_test = np.load('Mnist_set/Mnist_test_image.npy').reshape(-1,28,28,1)    # (10000, 28, 28, 1)
y_test = np.load('Mnist_set/Mnist_test_label.npy')                        # (10000,10)
y_test_dig = np.argmax(y_test,axis=1)    #  (10000,)

y_test_pred = resnet32.predict(x_test)   #  (10000,10)
y_test_pred_dig = np.argmax(y_test_pred,axis=1)     # (10000,)


# 计算FI
grad_K = tf.concat(axis=1,
    values=[
        tf.concat(axis=0, values=[K.flatten(b)[..., None] for b in K.gradients(resnet32.output[0, k], resnet32.input)])
    for k in range(n_classes)] )

iterate = K.function([resnet32.input], [grad_K, resnet32.output])

# compute training set's pic_FI under y=y_true and y=y_pred
FI_train = np.zeros(x_train.shape[0] )
FI_train_pred = np.zeros(x_train.shape[0] )

for i in range(x_train.shape[0]):

    grad, pred_P = iterate([x_train[i][None]])

    L0 = grad @ np.diag(((pred_P[0,]) ** 0.5 + epsilon) ** -1)

    f_grad = (grad[:, y_train_dig[i]] / (pred_P[0, y_train_dig[i]] + epsilon)).T  # shape=(1,...)
    f_grad_pred = (grad[:, y_train_pred_dig[i]] / (pred_P[0, y_train_pred_dig[i]] + epsilon)).T

    B0, D_L0, A0 = sp.linalg.svd(L0, full_matrices=False)
    rank_L0 = sum(D_L0 > epsilon)

    if rank_L0 > 0:
        B0 = B0[:, :rank_L0]
        A0 = np.diag(D_L0[:rank_L0]) @ A0[:rank_L0, :]

        U_A, D_0, _ = sp.linalg.svd(A0 @ A0.T, full_matrices=True)
        D_0_inv = np.diag(D_0 ** -1)
        D_0_inv_sqrt = np.diag(D_0 ** -0.5)

        U_0 = B0 @ U_A

        nabla_f = f_grad @ U_0 @ D_0_inv_sqrt
        nabla_f_pred = f_grad_pred @ U_0 @ D_0_inv_sqrt

        FI_train[i] = nabla_f @ nabla_f.T
        FI_train_pred[i] = nabla_f_pred @ nabla_f_pred.T

# compute test set's pic_FI under y=y_pred
FI_test_pred = np.zeros(x_test.shape[0] )

for i in range(x_test.shape[0]):
    grad, pred_P = iterate([x_test[i][None]])

    L0 = grad @ np.diag(((pred_P[0,]) ** 0.5 + epsilon) ** -1)

    f_grad_pred = (grad[:, y_test_pred_dig[i]] / (pred_P[0, y_test_pred_dig[i]] + epsilon)).T

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

        FI_test_pred[i] = nabla_f_pred @ nabla_f_pred.T

# 保存结果
workbook = xlwt.Workbook(encoding='utf-8')

worksheet_1 = workbook.add_sheet('Mnist_train_FI')
worksheet_2 = workbook.add_sheet('Mnist_train_pre_FI')
worksheet_3 = workbook.add_sheet('Mnist_test_pre_FI')

worksheet_1.write(0, 0, 'sample')      # sample id
worksheet_1.write(0, 1, 'FI_pic')
worksheet_1.write(0, 2, 'ture_class')
worksheet_1.write(0, 3, 'pred_class')
worksheet_1.write(0, 4, 'pred_pro')    # 4~13
worksheet_1.write(0, 14,'situation(1:right,0:wrong)')
for i in range(FI_train.shape[0]):     # 60000
    worksheet_1.write(i + 1, 0, i)
    worksheet_1.write(i + 1, 1, FI_train[i].astype(float))
    worksheet_1.write(i + 1, 2, y_train_dig[i].astype(float))
    worksheet_1.write(i + 1, 3, y_train_pred_dig[i].astype(float))
    for j in range(10):
        worksheet_1.write(i + 1, j + 4, y_train_pred[i][j].astype(float))
    if y_train_dig[i] == y_train_pred_dig[i]:
        worksheet_1.write(i + 1, 14, 1)
    else:
        worksheet_1.write(i + 1, 14, 0)

worksheet_2.write(0, 0, 'sample')       # sample id
worksheet_2.write(0, 1, 'FI_pic')
worksheet_2.write(0, 2, 'ture_class')
worksheet_2.write(0, 3, 'pred_class')
worksheet_2.write(0, 4, 'pred_pro')     # 4~13
worksheet_2.write(0, 14,'situation(1:right,0:wrong)')
for i in range(FI_train_pred.shape[0]):  # 60000
    worksheet_2.write(i + 1, 0, i)
    worksheet_2.write(i + 1, 1, FI_train_pred[i].astype(float))
    worksheet_2.write(i + 1, 2, y_train_dig[i].astype(float))
    worksheet_2.write(i + 1, 3, y_train_pred_dig[i].astype(float))
    for j in range(10):
        worksheet_2.write(i + 1, j + 4, y_train_pred[i][j].astype(float))
    if y_train_dig[i] == y_train_pred_dig[i]:
        worksheet_2.write(i + 1, 14, 1)
    else:
        worksheet_2.write(i + 1, 14, 0)

worksheet_3.write(0, 0, 'sample')      # sample id
worksheet_3.write(0, 1, 'FI_pic')
worksheet_3.write(0, 2, 'ture_class')
worksheet_3.write(0, 3, 'pred_class')
worksheet_3.write(0, 4, 'pred_pro')     # 4~13
worksheet_3.write(0, 14,'situation(1:right,0:wrong)')
for i in range(FI_test_pred.shape[0]):  # 60000
    worksheet_3.write(i + 1, 0, i)
    worksheet_3.write(i + 1, 1, FI_test_pred[i].astype(float))
    worksheet_3.write(i + 1, 2, y_test_dig[i].astype(float))
    worksheet_3.write(i + 1, 3, y_test_pred_dig[i].astype(float))
    for j in range(10):
        worksheet_3.write(i + 1, j + 4, y_test_pred[i][j].astype(float))
    if y_test_dig[i] == y_test_pred_dig[i]:
        worksheet_3.write(i + 1, 14, 1)
    else:
        worksheet_3.write(i + 1, 14, 0)

workbook.save('ResNet32_mnist_FI_pic.xls')





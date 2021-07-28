import numpy as np
import xlrd
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['font.sans-serif']='Times New Roman'
plt.rcParams['axes.unicode_minus']=False

situation = 0    # misjudged
class_list = [0,1,2,3,4,5,6,7,8,9]

# misjudgement relationship
def relationship_collect(sheet_id =1 ,situation = 0):
    if sheet_id == 1:
        sheet = sheet_train
    elif sheet_id == 2:
        sheet = sheet_train_pred
    elif sheet_id == 3:
        sheet = sheet_test_pred
    else:
        print('error! sheet_train:1 ,sheet_train_pred:2,sheet_test_pred: 3')
        return

    nrows = sheet.nrows
    relation_list = []

    for i in range(nrows - 1):
        if  sheet.cell_value(i + 1, 14) == situation:
            # [0:sample id，1:pic_FI value，2:true class，3:pred class, 4~13:pred pro, 14:situation(1:right,0:wrong)]
            relation_list.append([int(sheet.cell_value(i + 1, 2)),int(sheet.cell_value(i + 1, 3))])

    return relation_list


## Mnist
workbook =  xlrd.open_workbook('ResNet32_mnist_FI_pic.xls')

sheet_train = workbook.sheet_by_name('Mnist_train_FI')
sheet_train_pred = workbook.sheet_by_name('Mnist_train_pre_FI')
sheet_test_pred = workbook.sheet_by_name('Mnist_test_pre_FI')

# Mnist training set
sheet_id = 2

# misjudgement relationship of mnist training set
relation_list_train = relationship_collect(sheet_id = sheet_id ,situation = situation)

# misjudgement quantity of mnist training set
connect_quanity_array_train = np.zeros((10,10))
for i_relation_tuple in relation_list_train:
    connect_quanity_array_train[i_relation_tuple[0],i_relation_tuple[1]] +=1
a_quantity_train_m = np.array(connect_quanity_array_train).reshape((10,10))

# mnist test set
sheet_id = 3

# misjudgement relationship of mnist test set
relation_list_test = relationship_collect(sheet_id = sheet_id ,situation = situation)

# misjudgement quantity of mnist test set
connect_quanity_array_test = np.zeros((10,10))
for i_relation_tuple in relation_list_test:  #
    connect_quanity_array_test[i_relation_tuple[0],i_relation_tuple[1]] +=1
a_quantity_test_m = np.array(connect_quanity_array_test).reshape((10,10))



# Cifar10
workbook =  xlrd.open_workbook('ResNet32_cifar10_FI_pic.xls')

sheet_train = workbook.sheet_by_name('Cifar10_train_FI')
sheet_train_pred = workbook.sheet_by_name('Cifar10_train_pre_FI')
sheet_test_pred = workbook.sheet_by_name('Cifar10_test_pre_FI')

# training set
sheet_id = 2

# misjudgement relationship of cifar10 training set
relation_list_train= relationship_collect(sheet_id = sheet_id ,situation = situation)

# misjudgement quantity of cifar10 training set
connect_quanity_array_train = np.zeros((10,10))
for i_relation_tuple in relation_list_train:  #
    connect_quanity_array_train[i_relation_tuple[0],i_relation_tuple[1]] +=1
a_quantity_train_c = np.array(connect_quanity_array_train).reshape((10,10))

# test set
sheet_id = 3

# misjudgement relationship of cifar10 test set
relation_list_test = relationship_collect(sheet_id = sheet_id ,situation = situation)

# misjudgement quantity of cifar10 test set
connect_quanity_array_test = np.zeros((10,10))
for i_relation_tuple in relation_list_test:
    connect_quanity_array_test[i_relation_tuple[0],i_relation_tuple[1]] +=1
a_quantity_test_c = np.array(connect_quanity_array_test).reshape((10,10))


# heatmap of misjudgement
plt.figure(figsize=(20,6))

plt.subplot(1,4,1)
ax = sns.heatmap(a_quantity_train_m,annot=False, xticklabels= True, yticklabels= True, square=True, cmap="YlGnBu",cbar_kws={'shrink':0.5})
plt.ylabel('True class', fontsize = 20)
plt.xlabel('Pred class', fontsize = 20)
plt.yticks(class_list, fontsize = 20, rotation = 360, horizontalalignment='right')
plt.xticks(class_list, fontsize = 20, horizontalalignment='right')
plt.text(1.25,12.75,'(a) MNIST training set', fontsize = 20)
cax = plt.gcf().axes[-1]
cax.tick_params(labelsize=20)
cb = ax.collections[0].colorbar
cb.ax.tick_params(labelsize=20)

plt.subplot(1,4,2)
ax = sns.heatmap(a_quantity_test_m,annot=False,  xticklabels= True, yticklabels= True, square=True,  cmap="OrRd",cbar_kws={'shrink':0.5})
plt.ylabel('True class', fontsize = 20)
plt.xlabel('Pred class', fontsize = 20)
plt.yticks(class_list, fontsize = 20, rotation = 360, horizontalalignment='right')
plt.xticks(class_list, fontsize = 20, horizontalalignment='right')
plt.text(2,12.75,'(b) MNIST test set', fontsize = 20)
cax = plt.gcf().axes[-1]
cax.tick_params(labelsize=20)
cb = ax.collections[0].colorbar
cb.ax.tick_params(labelsize=20)


plt.subplot(1,4,3)
ax = sns.heatmap(a_quantity_train_c,annot=False, xticklabels= True, yticklabels= True, square=True, cmap="YlGnBu",cbar_kws={'shrink':0.5})
plt.ylabel('True class', fontsize = 20)
plt.xlabel('Pred class', fontsize = 20)
plt.yticks(class_list, fontsize = 20, rotation = 360, horizontalalignment='right')
plt.xticks(class_list, fontsize = 20, horizontalalignment='right')
plt.text(1,12.75,'(c) CIFAR10 training set', fontsize =20)
cax = plt.gcf().axes[-1]
cax.tick_params(labelsize=20)
cb = ax.collections[0].colorbar
cb.ax.tick_params(labelsize=20)

plt.subplot(1,4,4)
ax = sns.heatmap(a_quantity_test_c, annot=False, xticklabels= True, yticklabels= True, square=True, cmap="OrRd",cbar_kws={'shrink':0.5})
plt.ylabel('True class', fontsize = 20)
plt.xlabel('Pred class', fontsize = 20)
plt.yticks(class_list, fontsize = 20, rotation = 360, horizontalalignment='right')
plt.xticks(class_list, fontsize = 20, horizontalalignment='right')
plt.text(1.5,12.75,'(d) CIFAR10 test set', fontsize = 20)
cax = plt.gcf().axes[-1]
cax.tick_params(labelsize=20)
cb = ax.collections[0].colorbar
cb.ax.tick_params(labelsize=20)

plt.show()


# misjudgement relationship that Not less to a certain number of q_limit
q_limit = 10
a_quantity = np.array(a_quantity_train_c).reshape((10,10))
a = np.where(a_quantity>=q_limit)
list_of_i = a[0]
list_of_j = a[1]
info_train_sample = []
for k in range(len(list_of_i)):
    info_train_sample.append([list_of_i[k],list_of_j[k],a_quantity[list_of_i[k],list_of_j[k]]])
np.save('./adv_info/misjudgement_info/Cifar_(train)_%d'%q_limit,info_train_sample)  # [0：true_class, 1:pred_class, 2:mis_quantity]

q_limit = 5
a_quantity = np.array(a_quantity_train_m).reshape((10,10))
a = np.where(a_quantity>=q_limit)
list_of_i = a[0]
list_of_j = a[1]
info_train_sample = []
for k in range(len(list_of_i)):
    info_train_sample.append([list_of_i[k],list_of_j[k],a_quantity[list_of_i[k],list_of_j[k]]])
np.save('./adv_info/misjudgement_info/Mnist_(train)_%d'%q_limit,info_train_sample)  # [ture_class,target_class,mis_quantity]

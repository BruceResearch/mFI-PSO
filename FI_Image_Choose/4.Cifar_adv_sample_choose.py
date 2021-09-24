

import numpy as np
import xlrd

sheet_id = 3          # train: 1 or 2   test: 3
FI_below = 0.01
prob_y_target = 0.01
situation = 1             #  0:wrong，1：right
target_not_given = True

mis_info = np.load('./adv_info/misjudgement_info/Cifar_(train)_10.npy')
adv_target_dic = {}
for i in range(len(mis_info)):
    ture_class_i = int(mis_info[i][0])
    if  ture_class_i not in adv_target_dic.keys():
        adv_target_dic[ture_class_i] = []
    adv_target_dic[ture_class_i].append(int(mis_info[i][1]))
print(adv_target_dic)


# load excel
workbook =  xlrd.open_workbook('ResNet32_cifar10_FI_pic.xls')

# load sheet
sheet_train = workbook.sheet_by_name('Cifar10_train_FI')
sheet_train_pred = workbook.sheet_by_name('Cifar10_train_pre_FI')
sheet_test_pred = workbook.sheet_by_name('Cifar10_test_pre_FI')

def adv_select_without_target(sheet_id =1 ,FI_below=0.2, situation = 1,prob_y_target= 0.2):
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
    count_num = 0 
   
    adv_sample_dic = {  0:[],
                        1:[],
                        2:[],
                        3:[],
                        4:[],
                        5:[],
                        6:[],
                        7:[],
                        8:[],
                        9:[] }  
    sample_i_list = [[],[],[],[],[],[],[],[],[],[]]

    sample_j_dic = {}
    sample_j_list = []   
    for i in range(nrows - 1):  
        target_prob_list= [sheet.cell_value(i + 1, 4),
        sheet.cell_value(i + 1, 5),
        sheet.cell_value(i + 1, 6),
        sheet.cell_value(i + 1, 7),
        sheet.cell_value(i + 1, 8),
        sheet.cell_value(i + 1, 9),
        sheet.cell_value(i + 1, 10),
        sheet.cell_value(i + 1, 11),
        sheet.cell_value(i + 1, 12),
        sheet.cell_value(i + 1, 13)]
        secondmax = max(n for n in target_prob_list if n!=max(target_prob_list))

        j_target= int(target_prob_list.index(secondmax))
        j_target_col = int(4 + j_target)
        i_key = int(sheet.cell_value(i + 1, 2))
        if sheet.cell_value(i + 1, 14) == situation and \
            sheet.cell_value(i + 1, 1) >= FI_below and \
            sheet.cell_value(i + 1, j_target_col) >= prob_y_target and \
            sheet.cell_value(i + 1, 3) == i_key:
            sample_j_list = int(sheet.cell_value(i + 1, 0)) 

            count_num += 1
            a = {j_target: sample_j_list}
            #sample_j_dic[j_target] = sample_j_list  # j-target: [sample_id]

            sample_i_list[i_key].append(a)       # [j-target: [sample_id]]
            adv_sample_dic[i_key] = (sample_i_list[i_key])         # i_key:[j-target: [sample_id]]

    return adv_sample_dic,count_num

# find all the misjudgement
def adv_select(adv_target_dic, sheet_id =1 ,FI_below=0,situation = 0,prob_y_target= 0):
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
    count_num = 0        
    adv_sample_dic = {}  # key(ture):[key(y_target):sample_id]

    for i_key in adv_target_dic:
    
        if n_of_i_key == 1:   # only 1 y_target

            sample_i_list = []

            sample_j_dic = {}
            sample_j_list = []

            j_target = target_list_of_i[0]
            for i in range(nrows - 1):  
                j_target_col = int(4 + j_target)
             
                if sheet.cell_value(i + 1, 14) == situation and \
                        sheet.cell_value(i + 1, 1) >= FI_below and \
                        sheet.cell_value(i + 1, j_target_col) >= prob_y_target and \
                        sheet.cell_value(i + 1, 3) == i_key:
                    sample_j_list.append( int(sheet.cell_value(i + 1, 0)) )

                    count_num += 1

            sample_j_dic[j_target] = sample_j_list   # j-target: [sample_id]
            sample_i_list.append(sample_j_dic)       # [j-target: [sample_id]]
            adv_sample_dic[i_key] = sample_i_list    # i_key:[j-target: [sample_id]]

        else:  # y_target not only 1

            sample_i_dic = {}
            sample_i_list = []

            for j_target in target_list_of_i:

                sample_j_dic = {}
                sample_j_list = []

                for i in range(nrows - 1):  
                    j_target_col = int(4 + j_target)
      
                    if sheet.cell_value(i + 1, 14) == situation and \
                            sheet.cell_value(i + 1,j_target_col) >= prob_y_target and \
                            sheet.cell_value(i + 1, 3) == i_key:
                        sample_j_list.append(int(sheet.cell_value(i + 1, 0)))

                        count_num += 1

                sample_j_dic[j_target] = sample_j_list           # j-target: [sample_id]
                sample_i_list.append(sample_j_dic)     # [j-target: [sample_id]]
            adv_sample_dic[i_key] = sample_i_list       # i_key:[j-target: [sample_id]]

    return adv_sample_dic,count_num

if target_not_given == False:
    dic_cifar,count_num  = adv_select(adv_target_dic, sheet_id =sheet_id ,FI_below=FI_below,situation = situation,prob_y_target= prob_y_target)
else:
    dic_cifar,count_num  = adv_select_without_target(sheet_id =sheet_id ,FI_below=FI_below,situation = situation,prob_y_target= prob_y_target)
print(count_num)  

list_pic_num = []
for i_key in dic_cifar:   
    target_of_i_key_list = dic_cifar[i_key]
    for j_target_dic in target_of_i_key_list:
        for j_target_key in j_target_dic:      
        #print(j_target_key)              
            sample_of_j_target_list = j_target_dic[j_target_key]     
        #print(sample_of_j_target_list)
            
            list_pic_num.append([i_key,j_target_key,sample_of_j_target_list])
print(list_pic_num)
array_pic_num = np.array(list_pic_num)
print(array_pic_num.shape)         

np.save('./adv_info/sample_info/Cifar_sheet(%d)_situation(%d)_FI_below(%.2f)_pro_y_target(%.2f)_array.npy' % (sheet_id, situation, FI_below, prob_y_target), array_pic_num)

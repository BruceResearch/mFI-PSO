import random
from keras.models import *
import numpy as np
import matplotlib.pyplot as plt

K.set_learning_phase(0)

your_path = '/your_path_to_main_dir'

pop_size = 500
w = 0.6
c1 = 2
c2 = 2

p_erro = False

a_1 = 10000000
if p_erro:
    a_2 = 1
else:
    a_2 = 255

iter_num = 10
p_limit = 0.15

resnet32 = load_model(str(your_path) + '/FI_Image_Choose/my_resnet_32_mnist.h5')

# data
## train:
x_train = np.load(str(your_path) + '/FI_Image_Choose/Mnist_set/Mnist_train_image.npy').reshape(-1,28,28,1)    # (60000, 28 28, 1)
y_train = np.load(str(your_path) + '/FI_Image_Choose/Mnist_train_label.npy')                        # (60000, 10)

## test_set
x_test = np.load(str(your_path) + '/FI_Image_Choose/Mnist_set/Mnist_test_image.npy').reshape(-1,28,28,1)   # (10000, 28 28, 1)
y_test = np.load(str(your_path) + '/FI_Image_Choose/Mnist_set/Mnist_test_label.npy')

# position of the top m pixel FI
def posi_m_func(FI_pixel_x_adv,m):
    FI_pixel_ord = np.argsort(FI_pixel_x_adv[0])[::-1]
    FI_pointed_x_m = FI_pixel_ord[:m]
    FI_pointed_x_m = FI_pointed_x_m.reshape((m,1))
    return FI_pointed_x_m   #（m,1）

def disturb_limits(x_choose):
    disturb_limits = []
    for i in range(784):
        pixel_i_down =( 0 - x_choose[0][i])* p_limit
        pixel_i_up = (1 - x_choose[0][i])* p_limit
        disturb_limits.append([pixel_i_down,pixel_i_up])
    return  np.array(disturb_limits)   #(784, 2)


def org_pop_m(limits,posi_m, m ,pop_size):
    org_pop = np.zeros((pop_size,m))
    for i in range(pop_size):
        for j in range(m):
            posi_m_j = posi_m[j][0]
            org_pop[i][j] = random.uniform(limits[posi_m_j][0],limits[posi_m_j][1])
    return org_pop    #(pop_size,m)


# compute fitness
def calculate_object_score(m,posi_m,target_class,correct_class,x_pop,x_adv,p_erro,a1=1000000,a2=1):  
    results = np.zeros((pop_size,2+m))  

    for i in range(pop_size):

        # 计算aim1的分值
        x_adv_flatten = x_adv.reshape((1,-1))
        pixel_adv = np.zeros((1,784))
        for j in range(784):
            pixel_adv[0][j] = x_adv_flatten[0][j]
            if j in posi_m:
                index_j_in_m =  np.where(j == posi_m)   
                pixel_adv[0][j] = x_adv_flatten[0][j] + x_pop[i][index_j_in_m[0][0]]   

        
        y_pre = resnet32.predict(pixel_adv.reshape((1,28,28,1)) ) # (1, 10)

        k_sort = np.argsort(y_pre)  #(1,10)

        k1_pre = k_sort[0][-1]
        k2_pre = k_sort[0][-2]

        if target_class != correct_class:
            if k1_pre != target_class:  
                if p_erro:
                    aim_1_score_i = np.abs(y_pre[0][target_class] - p_erro)
                else:
                    aim_1_score_i = np.abs(y_pre[0][k1_pre] - y_pre[0][target_class])
                # results[i][1] = 0

            else:  # k1_pre == target_class
                if p_erro:
                    aim_1_score_i = np.round(np.abs(y_pre[0][target_class] - p_erro) * 0.1, 3)
                else:
                    aim_1_score_i = np.round(np.abs(y_pre[0][target_class] - y_pre[0][k2_pre]) * 0.001,  6)  
                # aim_1_score_i = 0
                results[i][1] = 1  
        else:
            if k1_pre == correct_class:  
                if p_erro:
                    aim_1_score_i = np.abs(y_pre[0][k2_pre] - p_erro)
                else:
                    aim_1_score_i = np.abs(y_pre[0][k1_pre] - y_pre[0][k2_pre])

            else:
                if p_erro:
                    aim_1_score_i = np.round(np.abs(y_pre[0][k1_pre] - p_erro) * 0.1, 3)
                else:
                    aim_1_score_i = np.round(np.abs(y_pre[0][k1_pre] - y_pre[0][correct_class]) * 0.001, 6)
                # aim_1_score_i = 0
                results[i][1] = 1

       
        aim_2_score_i = 0
        for m_i in range(m):
            aim_2_score_i += np.abs(x_pop[i][m_i])

        
        Q_score_i = a1 * aim_1_score_i + a2 * aim_2_score_i

        results[i][0] = Q_score_i
        for m_i in range(m):
            results[i][m_i+2] = x_pop[i][m_i]

    return results 

def personal_best(p_best_result,result_in_iter):
    for i in range(pop_size):
        if p_best_result[0][0]>= result_in_iter[i][0] :
            p_best_result[i] = result_in_iter[i]
    return p_best_result   #（1，2+m）

def iter_best(result_in_iter):
    Q_score_i = result_in_iter[:,0].reshape((1,-1))
    best_posi_i = np.argmin(Q_score_i)
    best_result_i = result_in_iter[best_posi_i].reshape((1,-1))
    return best_result_i   # (1,2+m)

def global_best(g_best_result,best_result_in_iter):
    if g_best_result[0][0] >= best_result_in_iter[0][0]:
        g_best_result = best_result_in_iter
    return g_best_result     # (1,2+m)

def evolve(m,posi_m,w,v,c1,c2,x_pop, p_best_result ,g_best_result,dist_x_limits):
    p_best = p_best_result[:,2:].reshape((pop_size,m))
    g_best = g_best_result[0][2:].reshape((1,m))

    r1 = np.random.rand(pop_size, m)
    r2 = np.random.rand(pop_size, m)

    # update
    v = w * v + c1 * r1 * (p_best - x_pop) + c2 * r2 * (g_best - x_pop)

    for i in range(pop_size):
        for j in range(m):
            m_idex = posi_m[j][0]
            v[i][j]=np.clip(v[i][j],dist_x_limits[m_idex][0]*0.05,dist_x_limits[m_idex][1]*0.05)

    x_pop_new = v + x_pop

    for i in range(pop_size):
        for j in range(m):
            m_idex = posi_m[j][0]
            x_pop_new[i][j]=np.clip(x_pop_new[i][j],dist_x_limits[m_idex][0],dist_x_limits[m_idex][1])

    return x_pop_new

def classify(img,probs,img_name,color = True ,correct_class=None, target_class=None,save_flag = False):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 8))
    fig.sca(ax1)
    if color:
        ax1.imshow(img)
    else:
        im1 = ax1.imshow(img, cmap='Greys_r')
        #plt.colorbar(im1, ticks=[-255,-100, 0, 100,255], shrink=0.35)
    fig.sca(ax1)

    p = probs  #
    topk = list(np.argsort(p)[::-1])   
    topprobs = p[topk]                 
    barlist = ax2.bar(range(10), topprobs)
    if target_class in topk:
        barlist[topk.index(target_class)].set_color('r')
    if correct_class in topk:
        barlist[topk.index(correct_class)].set_color('g')
    plt.sca(ax2)
    plt.ylim([0, 1.1])
    plt.xticks(range(10),[i for i in topk],rotation='vertical') 
    fig.subplots_adjust(bottom=0.2)

    if save_flag:
        plt.savefig('%s.png'%(str(img_name)))
    plt.show()

def PSO_critical_point(pic_num,pixel_FI_array, m ,correct_class,target_class ,sheet_id =2 ):

    if sheet_id == 3:
        x_adv = x_test[int(pic_num)]    #  (28, 28, 1)
        y_adv = y_test[int(pic_num)]
    else:
         x_adv = x_train[int(pic_num)]   # (28, 28, 1)
         y_adv = y_train[int(pic_num)]


    posi_m = posi_m_func(pixel_FI_array,m)      # (m,1)

    
    dist_x_limits = disturb_limits(x_adv.reshape(1,-1))          
    print('position_of_m and perturbation range are：')
    for i in range(m):
        posi_m_i = posi_m[i][0]
        print('(',i,'/',m,'): position: ',posi_m_i,'-->limit:',dist_x_limits[posi_m_i][0],dist_x_limits[posi_m_i][1])

    
    x_pop_temp = org_pop_m(dist_x_limits,posi_m,m,pop_size)       

    v = np.random.rand(pop_size, m) * 0.1 

    g_best_result = np.zeros((1,2+m)) + 10000000
    p_best_result = np.zeros((pop_size,2+m)) +10000000

    for i in range(iter_num):
        
        result_in_iter = calculate_object_score(m,posi_m,target_class,correct_class,x_pop_temp, x_adv,p_erro,a1=a_1,a2=a_2)

        
        p_best_result = personal_best(p_best_result,result_in_iter)

        best_result_in_iter = iter_best(result_in_iter)

        g_best_result = global_best(g_best_result, best_result_in_iter)
        print('step_'+str(i)+'_global_best_result',g_best_result[0][0:2])

        x_pop_temp = evolve(m,posi_m, w, v, c1, c2,x_pop_temp ,p_best_result, best_result_in_iter,dist_x_limits)

    result_final = calculate_object_score(m,posi_m,target_class,correct_class,x_pop_temp, x_adv,p_erro,a1=a_1,a2=a_2)

    best_result_final = iter_best(result_final)

    g_best_result_final = global_best(g_best_result, best_result_final)
    print('final_result:', g_best_result_final[0][:2])

 
    x_adv_final_info_list = []

    x_adv_final_info_list.append(g_best_result_final[0][1])   #   #    [0 :   situation,  1~784: adv_final ]

   
    x_adv_flatten = x_adv.reshape((1, -1))
    a = x_adv_flatten
    x_adv_final = np.zeros((1, 784))

    for j in range(784):

        x_adv_final[0][j] = x_adv_flatten[0][j]
        if j in posi_m:
            index_j_in_m = np.where(j == posi_m)  
            result_index = index_j_in_m[0][0] + 2
            x_adv_final[0][j] = x_adv_flatten[0][j] + g_best_result_final[0][result_index]  
            x_adv_final[0][j] = np.clip(x_adv_final[0][j], a[0][j] - p_limit, a[0][j]+ p_limit)
            x_adv_final[0][j] = np.clip(x_adv_final[0][j], 0, 1)
        x_adv_final_info_list.append(x_adv_final[0][j])   #  #    [0 :   situation,  1~784: adv_final ]


    
    y_adv_pre = resnet32.predict(x_adv_final.reshape((1,28,28,1)))
    print('prediction after perturbation：',y_adv_pre)
    if np.argmax(y_adv_pre, axis =1) == correct_class:
        y_adv_pre1 = y_adv_pre
        p_shape = y_adv_pre1.shape
        print(p_shape)
        y_adv_pre1[0,np.argmax(y_adv_pre1, axis =1)] = np.min(y_adv_pre, axis =1)
        target_class = np.argmax(y_adv_pre1, axis =1)
        for i in range(iter_num * 4):
           
            
            result_in_iter = calculate_object_score(m,posi_m,target_class,correct_class,x_pop_temp, x_adv,p_erro,a1=a_1,a2=a_2)

           
            p_best_result = personal_best(p_best_result,result_in_iter)

            best_result_in_iter = iter_best(result_in_iter)

            g_best_result = global_best(g_best_result, best_result_in_iter)
            print('step_'+str(i)+'_global_best_result',g_best_result[0][0:2])

            x_pop_temp = evolve(m,posi_m, w, v, c1, c2,x_pop_temp ,p_best_result, best_result_in_iter,dist_x_limits)

        result_final = calculate_object_score(m,posi_m,target_class,correct_class,x_pop_temp, x_adv,p_erro,a1=a_1,a2=a_2)

        best_result_final = iter_best(result_final)

        g_best_result_final = global_best(g_best_result, best_result_final)
        print('final_result:', g_best_result_final[0][:2])

      
        x_adv_final_info_list = []

        x_adv_final_info_list.append(g_best_result_final[0][1])   #   #    [0 :   situation,  1~784: adv_final ]

       

        for j in range(784):

            if j in posi_m:
                index_j_in_m = np.where(j == posi_m)  
                result_index = index_j_in_m[0][0] + 2
                x_adv_final[0][j] = x_adv_final[0][j] + g_best_result_final[0][result_index]  
                x_adv_final[0][j] = np.clip(x_adv_final[0][j], a[0][j] - p_limit, a[0][j]+ p_limit)
                x_adv_final[0][j] = np.clip(x_adv_final[0][j], 0, 1)


            x_adv_final_info_list.append(x_adv_final[0][j])   #  #    [0 :   situation,  1~784: adv_final ]
            
        y_adv_pre = resnet32.predict(x_adv_final.reshape((1,28,28,1)))
        print('prediction after perturbation：',y_adv_pre)

    
    img_name_adv = road_str_adv
    classify(x_adv_final.reshape((28,28)), y_adv_pre[0], img_name_adv, color=False, correct_class=correct_class, target_class=target_class,save_flag=True)


    x_adv_info = np.array(x_adv_final_info_list).reshape((1,-1))    #  #    [0 :   situation,  1~3072: adv_final ]

    print( x_adv_info.shape)

    return x_adv_final, y_adv
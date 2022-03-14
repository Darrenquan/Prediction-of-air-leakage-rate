# import tensorflow.compat.v1 as tf
# tf.compat.v1.disable_v2_behavior()
# import tensorflow as tf
#第一次用上面的语句跑的时候还好好的，再跑就报错了
import tensorflow as tf
tf = tf.compat.v1
tf.disable_v2_behavior()


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler


def function(x1,y1,x2,y2,W):#  W是神经网络的W权重，根据这个权重设置神经网络




    #定义激活函数
    activation_function=tf.nn.relu
    #输入输出数据集
    xs=tf.placeholder(tf.float32,[None,None])
    ys=tf.placeholder(tf.float32,[None,None])




    #设计bp神经网络，三层，13,3,1
    weights_1=tf.Variable(W[0,:,:],tf.float32)
    biases_1=tf.Variable(tf.zeros([1,3])+0.1,tf.float32)
    wx_plus_b_1=tf.matmul( xs, tf.cast(weights_1,tf.float32))+biases_1
    outputs_1=activation_function(wx_plus_b_1)


    weights_2=tf.Variable(W[1,0:3,:],tf.float32)
    biases_2=tf.Variable(tf.zeros([1,3])+0.1,tf.float32)
    wx_plus_b_2=tf.matmul(outputs_1 , tf.cast(weights_2,tf.float32))+biases_2
    outputs_2=activation_function(wx_plus_b_2)


    w3=W[2,0:3,0].reshape(3,1)
    weights_3=tf.Variable(w3,tf.float32)
    biases_3=tf.Variable(0.1,tf.float32)

    wx_plus_b_3=tf.matmul(outputs_2,tf.cast(weights_3,tf.float32))+biases_3



    #预测输出结果
    prediction=wx_plus_b_3      #看来这里的数据就用行向量来输入输出

    #定义损失函数
    loss=tf.reduce_mean(tf.reduce_sum(tf.square(y1-prediction),reduction_indices=[1]))

    #梯度下降法训练
    train_step=tf.train.GradientDescentOptimizer(0.1).minimize(loss)

    #初始化变量
    init=tf.global_variables_initializer()

    #执行会话，开始训练模型
    print("开始")
    with tf.Session() as sess:
        sess.run(init)
        for i in range (1000):
            sess.run(train_step,feed_dict={ xs:x1  , ys:y1 })

#为什么损失函数喂入x2,y2就不行？QAQ
        end_loss=sess.run(loss,feed_dict={xs:x1,ys:y1})
        print(end_loss)
# print(sess.run(prediction,feed_dict={xs:x2}))
        print("结束")
    return end_loss


#导入数据集
data=load_boston()
data_pd=pd.DataFrame(data.data,columns=data.feature_names)
data_pd["price"]=data.target




#dataframe导入numpy
x=np.array(data_pd.loc[:,'CRIM':'LSTAT'])

y=np.array(data_pd.loc[:,'price'])

y.shape=(506,1)
#训练集测试集
x_train,x_test,y_train,y_test=train_test_split(x,y , test_size=0.1 )
#数据标准化
SC=StandardScaler()
x_train=SC.fit_transform(x_train)
y_train=SC.fit_transform(y_train)
x_test=SC.fit_transform(x_test)
y_test=SC.fit_transform(y_test)



#粒子数量num
num = 3

#粒子位置矩阵的形状
num_x = 3
num_y = 13
num_z = 3

#p为粒子位置矩阵，初始化为标准正态分布
p = np.random.randn(num,num_x,num_y,num_z)

#初始化粒子速度,以标准正态分布随机初始化
v = np.random.randn(num,num_x,num_y,num_z)

#个体最佳位置
good_p = np.array(p, copy=True)

#全局最佳位置
best_p = np.zeros((num_x, num_y, num_z))

#每次粒子移动后所计算出新的目标函数值
new_y = np.zeros(num)

#粒子个体历史最优值
good_y = np.zeros(num)

#粒子群体历史最优值
best_y = 0

#计算出初始粒子群的目标函数值
for i in range(num):
    good_y[i] = function(x_train, y_train, x_test, y_test, p[i, :, :, :])

#目标函数返回值是误差，那么最小的就是最优的
best_y = min(good_y)

#确定初始时最优位置
best_p = p[np.argmin(good_y), :, :, :]

#设置最大迭代次数
max_iter = 10

#开始迭代
for i in range(max_iter):

    #速度更新公式
    v = random.random() * v + 2.4 * random.random() * (best_p - p) + 1.7 * random.random() * ( good_p - p )

    #粒子位置更新
    p = p + v

    #计算每个粒子到达新位置后所得到的目标函数值
    for i in range(num):
        new_y[i] = function(x_train, y_train, x_test, y_test, p[i, :, :, :])

    #更新全局最优
    if min(new_y) < best_y:
        best_y = min(new_y)
        best_p = p[np.argmin(new_y), :, :, :]

    #更新个体历史最优
    for i in range(num):
        if new_y[i] < good_y[i]:
            good_y[i] = new_y[i]
            good_p[i, :, :, :] = p[i, :, :, :]  # 当对切片修改时，原始numpy数据也修改


print("结束")
print('目标函数最优值：',best_y)
print('此时的粒子位置：',best_p)


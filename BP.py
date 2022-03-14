import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import sys

def sigmoid(x):
    # 第一层到第二层的激活函数
    return 1 / (1 + np.exp(-x))


def deriv_sigmoid(x):
    # 第一层到第二层的激活函数的求导函数
    fx = sigmoid(x)
    return fx * (1 - fx)


def mse_loss(y_true, y_pred):
    # 使用方差作为损失函数
    return ((y_true - y_pred) ** 2).mean()


class OurNeuralNetwork:

    def __init__(self):
        # 第一层到第二层的函数
        self.w11 = np.random.normal()
        self.w12 = np.random.normal()
        self.w13 = np.random.normal()
        self.w14 = np.random.normal()
        self.w15 = np.random.normal()
        self.w16 = np.random.normal()
        self.w17 = np.random.normal()
        self.w18 = np.random.normal()
        self.w19 = np.random.normal()
        self.w110 = np.random.normal()
        self.w111 = np.random.normal()
        self.w112 = np.random.normal()
        self.w113 = np.random.normal()
        self.w114 = np.random.normal()
        self.w115 = np.random.normal()

        self.w21 = np.random.normal()
        self.w22 = np.random.normal()
        self.w23 = np.random.normal()
        self.w24 = np.random.normal()
        self.w25 = np.random.normal()
        self.w26 = np.random.normal()
        self.w27 = np.random.normal()
        self.w28 = np.random.normal()
        self.w29 = np.random.normal()
        self.w210 = np.random.normal()
        self.w211 = np.random.normal()
        self.w212 = np.random.normal()
        self.w213 = np.random.normal()
        self.w214 = np.random.normal()
        self.w215 = np.random.normal()
        
        self.w31 = np.random.normal()
        self.w32 = np.random.normal()
        self.w33 = np.random.normal()
        self.w34 = np.random.normal()
        self.w35 = np.random.normal()
        self.w36 = np.random.normal()
        self.w37 = np.random.normal()
        self.w38 = np.random.normal()
        self.w39 = np.random.normal()
        self.w310 = np.random.normal()
        self.w311 = np.random.normal()
        self.w312 = np.random.normal()
        self.w313 = np.random.normal()
        self.w314 = np.random.normal()
        self.w315 = np.random.normal()
        
        self.w41 = np.random.normal()
        self.w42 = np.random.normal()
        self.w43 = np.random.normal()
        self.w44 = np.random.normal()
        self.w45 = np.random.normal()
        self.w46 = np.random.normal()
        self.w47 = np.random.normal()
        self.w48 = np.random.normal()
        self.w49 = np.random.normal()
        self.w410 = np.random.normal()
        self.w411 = np.random.normal()
        self.w412 = np.random.normal()
        self.w413 = np.random.normal()
        self.w414 = np.random.normal()
        self.w415 = np.random.normal()

        # 第二层到第三层的函数
        self.w1 = np.random.normal()
        self.w2 = np.random.normal()
        self.w3 = np.random.normal()
        self.w4 = np.random.normal()
        # 截距项，Biases
        self.b1 = np.random.normal()
        self.b2 = np.random.normal()
        self.b3 = np.random.normal()
        self.b4 = np.random.normal()
        self.b5 = np.random.normal()

    def feedforward(self, x):
        # 前向传播学习
        h1 = sigmoid(self.w11 * x[0] + self.w12 * x[1] + self.w13 * x[2] + self.w14 * x[3] + self.w15 * x[4] + self.w16 * x[5] + self.w17 * x[6] + self.w18 * x[7] + self.w19 * x[8] + self.w110 * x[9] + self.w111 * x[10] + self.w112 * x[11] + self.w113 * x[12]+ self.w114 * x[13] + self.w115 * x[14] + self.b1)
        h2 = sigmoid(self.w21 * x[0] + self.w22 * x[1] + self.w23 * x[2] + self.w24 * x[3] + self.w25 * x[4] + self.w26 * x[5] + self.w27 * x[6] + self.w28 * x[7] + self.w29 * x[8] + self.w210 * x[9] + self.w211 * x[10] + self.w212 * x[11] + self.w213 * x[12]+ self.w214 * x[13] + self.w215 * x[14] + self.b2)
        h3 = sigmoid(self.w31 * x[0] + self.w32 * x[1] + self.w33 * x[2] + self.w34 * x[3] + self.w35 * x[4] + self.w36 * x[5] + self.w37 * x[6] + self.w38 * x[7] + self.w39 * x[8] + self.w310 * x[9] + self.w311 * x[10] + self.w312 * x[11] + self.w313 * x[12]+ self.w314 * x[13] + self.w315 * x[14] + self.b3)
        h4 = sigmoid(self.w41 * x[0] + self.w42 * x[1] + self.w43 * x[2] + self.w44 * x[3] + self.w45 * x[4] + self.w46 * x[5] + self.w47 * x[6] + self.w48 * x[7] + self.w49 * x[8] + self.w410 * x[9] + self.w411 * x[10] + self.w412 * x[11] + self.w413 * x[12]+ self.w414 * x[13] + self.w415 * x[14] + self.b4)
        
        o1 = self.w1 * h1 + self.w2 * h2 + self.w3 * h3 + self.w4 * h4 + self.b5
        return o1
    #训练函数
    def train(self, data, all_y_trues):
        learn_rate = 0.001  # 学习率
        epochs = 3000  # 训练的次数
        # 画图数据
        self.loss = np.zeros(100)
        self.sum = 0
        # 开始训练
        for epoch in range(epochs):
            for x, y_true in zip(data, all_y_trues):
                # 计算h1
                h1 = sigmoid(
                    self.w11 * x[0] + self.w12 * x[1] + self.w13 * x[2] + self.w14 * x[3] + self.w15 * x[4] + self.w16 *
                    x[5] + self.w17 * x[6] + self.w18 * x[7] + self.w19 * x[8] + self.w110 * x[9] + self.w111 * x[
                        10] + self.w112 * x[11] + self.w113 * x[12] + self.w114 * x[13] + self.w115 * x[14] + self.b1)
                # 计算h2
                h2 = sigmoid(
                    self.w21 * x[0] + self.w22 * x[1] + self.w23 * x[2] + self.w24 * x[3] + self.w25 * x[4] + self.w26 *
                    x[5] + self.w27 * x[6] + self.w28 * x[7] + self.w29 * x[8] + self.w210 * x[9] + self.w211 * x[
                        10] + self.w212 * x[11] + self.w213 * x[12] + self.w214 * x[13] + self.w215 * x[14] + self.b2)

                h3 = sigmoid(
                    self.w31 * x[0] + self.w32 * x[1] + self.w33 * x[2] + self.w34 * x[3] + self.w35 * x[4] + self.w36 *
                    x[5] + self.w37 * x[6] + self.w38 * x[7] + self.w39 * x[8] + self.w310 * x[9] + self.w311 * x[
                        10] + self.w312 * x[11] + self.w313 * x[12] + self.w314 * x[13] + self.w315 * x[14] + self.b3)

                h4 = sigmoid(
                    self.w41 * x[0] + self.w42 * x[1] + self.w43 * x[2] + self.w44 * x[3] + self.w45 * x[4] + self.w46 *
                    x[5] + self.w47 * x[6] + self.w48 * x[7] + self.w49 * x[8] + self.w410 * x[9] + self.w411 * x[
                        10] + self.w412 * x[11] + self.w413 * x[12] + self.w414 * x[13] + self.w415 * x[14] + self.b4)

                #计算输出节点
                y_pred = self.w1 * h1 + self.w2 * h2 + self.w3 * h3 + self.w4 * h4 + self.b5
                # 反向传播计算导数
                d_L_d_ypred = -2 * (y_true - y_pred)
                d_ypred_d_w1 = h1
                d_ypred_d_w2 = h2
                d_ypred_d_w3 = h3
                d_ypred_d_w4 = h4
                d_ypred_d_b5 = 0
                d_ypred_d_h1 = self.w1
                d_ypred_d_h2 = self.w2
                d_ypred_d_h3 = self.w3
                d_ypred_d_h4 = self.w4

                sum_1 = self.w11 * x[0] + self.w12 * x[1] + self.w13 * x[2] + self.w14 * x[3] + self.w15 * x[4] + self.w16 * x[5] + self.w17 * x[6] + self.w18 * x[7] + self.w19 * x[8] + self.w110 * x[9] + self.w111 * x[10] + self.w112 * x[11] + self.w113 * x[12] + self.w114 * x[13] + self.w115 * x[14] + self.b1
                d_h1_d_w11 = x[0] * deriv_sigmoid(sum_1)
                d_h1_d_w12 = x[1] * deriv_sigmoid(sum_1)
                d_h1_d_w13 = x[2] * deriv_sigmoid(sum_1)
                d_h1_d_w14 = x[3] * deriv_sigmoid(sum_1)
                d_h1_d_w15 = x[4] * deriv_sigmoid(sum_1)
                d_h1_d_w16 = x[5] * deriv_sigmoid(sum_1)
                d_h1_d_w17 = x[6] * deriv_sigmoid(sum_1)
                d_h1_d_w18 = x[7] * deriv_sigmoid(sum_1)
                d_h1_d_w19 = x[8] * deriv_sigmoid(sum_1)
                d_h1_d_w110 = x[9] * deriv_sigmoid(sum_1)
                d_h1_d_w111 = x[10] * deriv_sigmoid(sum_1)
                d_h1_d_w112 = x[11] * deriv_sigmoid(sum_1)
                d_h1_d_w113 = x[12] * deriv_sigmoid(sum_1)
                d_h1_d_w114 = x[13] * deriv_sigmoid(sum_1)
                d_h1_d_w115 = x[14] * deriv_sigmoid(sum_1)
                d_h1_d_b1 = deriv_sigmoid(sum_1)

                sum_2 = self.w21 * x[0] + self.w22 * x[1] + self.w23 * x[2] + self.w24 * x[3] + self.w25 * x[4] + self.w26 * x[5] + self.w27 * x[6] + self.w28 * x[7] + self.w29 * x[8] + self.w210 * x[9] + self.w211 * x[10] + self.w212 * x[11] + self.w213 * x[12]+ self.w214 * x[13] + self.w215 * x[14] + self.b2
                d_h1_d_w21 = x[0] * deriv_sigmoid(sum_2)
                d_h1_d_w22 = x[1] * deriv_sigmoid(sum_2)
                d_h1_d_w23 = x[2] * deriv_sigmoid(sum_2)
                d_h1_d_w24 = x[3] * deriv_sigmoid(sum_2)
                d_h1_d_w25 = x[4] * deriv_sigmoid(sum_2)
                d_h1_d_w26 = x[5] * deriv_sigmoid(sum_2)
                d_h1_d_w27 = x[6] * deriv_sigmoid(sum_2)
                d_h1_d_w28 = x[7] * deriv_sigmoid(sum_2)
                d_h1_d_w29 = x[8] * deriv_sigmoid(sum_2)
                d_h1_d_w210 = x[9] * deriv_sigmoid(sum_2)
                d_h1_d_w211 = x[10] * deriv_sigmoid(sum_2)
                d_h1_d_w212 = x[11] * deriv_sigmoid(sum_2)
                d_h1_d_w213 = x[12] * deriv_sigmoid(sum_2)
                d_h1_d_w214 = x[13] * deriv_sigmoid(sum_2)
                d_h1_d_w215 = x[14] * deriv_sigmoid(sum_2)
                d_h1_d_b2 = deriv_sigmoid(sum_2)

                sum_3 = self.w31 * x[0] + self.w32 * x[1] + self.w33 * x[2] + self.w34 * x[3] + self.w35 * x[4] + self.w36 * x[5] + self.w37 * x[6] + self.w38 * x[7] + self.w39 * x[8] + self.w310 * x[9] + self.w311 * x[10] + self.w312 * x[11] + self.w313 * x[12]+ self.w314 * x[13] + self.w315 * x[14] + self.b3

                d_h1_d_w31 = x[0] * deriv_sigmoid(sum_3)
                d_h1_d_w32 = x[1] * deriv_sigmoid(sum_3)
                d_h1_d_w33 = x[2] * deriv_sigmoid(sum_3)
                d_h1_d_w34 = x[3] * deriv_sigmoid(sum_3)
                d_h1_d_w35 = x[4] * deriv_sigmoid(sum_3)
                d_h1_d_w36 = x[5] * deriv_sigmoid(sum_3)
                d_h1_d_w37 = x[6] * deriv_sigmoid(sum_3)
                d_h1_d_w38 = x[7] * deriv_sigmoid(sum_3)
                d_h1_d_w39 = x[8] * deriv_sigmoid(sum_3)
                d_h1_d_w310 = x[9] * deriv_sigmoid(sum_3)
                d_h1_d_w311 = x[10] * deriv_sigmoid(sum_3)
                d_h1_d_w312 = x[11] * deriv_sigmoid(sum_3)
                d_h1_d_w313 = x[12] * deriv_sigmoid(sum_3)
                d_h1_d_w314 = x[13] * deriv_sigmoid(sum_3)
                d_h1_d_w315 = x[14] * deriv_sigmoid(sum_3)
                d_h1_d_b3 = deriv_sigmoid(sum_3)

                sum_4 = self.w41 * x[0] + self.w42 * x[1] + self.w43 * x[2] + self.w44 * x[3] + self.w45 * x[4] + self.w46 * x[5] + self.w47 * x[6] + self.w48 * x[7] + self.w49 * x[8] + self.w410 * x[9] + self.w411 * x[10] + self.w412 * x[11] + self.w413 * x[12]+ self.w414 * x[13] + self.w415 * x[14] + self.b4

                d_h1_d_w41 = x[0] * deriv_sigmoid(sum_4)
                d_h1_d_w42 = x[1] * deriv_sigmoid(sum_4)
                d_h1_d_w43 = x[2] * deriv_sigmoid(sum_4)
                d_h1_d_w44 = x[3] * deriv_sigmoid(sum_4)
                d_h1_d_w45 = x[4] * deriv_sigmoid(sum_4)
                d_h1_d_w46 = x[5] * deriv_sigmoid(sum_4)
                d_h1_d_w47 = x[6] * deriv_sigmoid(sum_4)
                d_h1_d_w48 = x[7] * deriv_sigmoid(sum_4)
                d_h1_d_w49 = x[8] * deriv_sigmoid(sum_4)
                d_h1_d_w410 = x[9] * deriv_sigmoid(sum_4)
                d_h1_d_w411 = x[10] * deriv_sigmoid(sum_4)
                d_h1_d_w412 = x[11] * deriv_sigmoid(sum_4)
                d_h1_d_w413 = x[12] * deriv_sigmoid(sum_4)
                d_h1_d_w414 = x[13] * deriv_sigmoid(sum_4)
                d_h1_d_w415 = x[14] * deriv_sigmoid(sum_4)
                d_h1_d_b4 = deriv_sigmoid(sum_4)

                # 梯度下降法
                self.w11 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w11
                self.w12 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w12
                self.w13 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w13
                self.w14 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w14
                self.w15 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w15
                self.w16 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w16
                self.w17 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w17
                self.w18 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w18
                self.w19 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w19
                self.w110 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w110
                self.w111 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w111
                self.w112 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w112
                self.w113 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w113
                self.w114 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w114
                self.w115 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w115

                self.b1 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_b1

                self.w21 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h1_d_w21
                self.w22 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h1_d_w22
                self.w23 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h1_d_w23
                self.w24 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h1_d_w24
                self.w25 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h1_d_w25
                self.w26 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h1_d_w26
                self.w27 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h1_d_w27
                self.w28 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h1_d_w28
                self.w29 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h1_d_w29
                self.w210 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h1_d_w210
                self.w211 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h1_d_w211
                self.w212 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h1_d_w212
                self.w213 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h1_d_w213
                self.w214 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h1_d_w214
                self.w215 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h1_d_w215

                self.b2 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h1_d_b2

                self.w31 -= learn_rate * d_L_d_ypred * d_ypred_d_h3 * d_h1_d_w31
                self.w32 -= learn_rate * d_L_d_ypred * d_ypred_d_h3 * d_h1_d_w32
                self.w33 -= learn_rate * d_L_d_ypred * d_ypred_d_h3 * d_h1_d_w33
                self.w34 -= learn_rate * d_L_d_ypred * d_ypred_d_h3 * d_h1_d_w34
                self.w35 -= learn_rate * d_L_d_ypred * d_ypred_d_h3 * d_h1_d_w35
                self.w36 -= learn_rate * d_L_d_ypred * d_ypred_d_h3 * d_h1_d_w36
                self.w37 -= learn_rate * d_L_d_ypred * d_ypred_d_h3 * d_h1_d_w37
                self.w38 -= learn_rate * d_L_d_ypred * d_ypred_d_h3 * d_h1_d_w38
                self.w39 -= learn_rate * d_L_d_ypred * d_ypred_d_h3 * d_h1_d_w39
                self.w310 -= learn_rate * d_L_d_ypred * d_ypred_d_h3 * d_h1_d_w310
                self.w311 -= learn_rate * d_L_d_ypred * d_ypred_d_h3 * d_h1_d_w311
                self.w312 -= learn_rate * d_L_d_ypred * d_ypred_d_h3 * d_h1_d_w312
                self.w313 -= learn_rate * d_L_d_ypred * d_ypred_d_h3 * d_h1_d_w313
                self.w314 -= learn_rate * d_L_d_ypred * d_ypred_d_h3 * d_h1_d_w314
                self.w315 -= learn_rate * d_L_d_ypred * d_ypred_d_h3 * d_h1_d_w315

                self.b3 -= learn_rate * d_L_d_ypred * d_ypred_d_h3 * d_h1_d_b3

                self.w41 -= learn_rate * d_L_d_ypred * d_ypred_d_h4 * d_h1_d_w41
                self.w42 -= learn_rate * d_L_d_ypred * d_ypred_d_h4 * d_h1_d_w42
                self.w43 -= learn_rate * d_L_d_ypred * d_ypred_d_h4 * d_h1_d_w43
                self.w44 -= learn_rate * d_L_d_ypred * d_ypred_d_h4 * d_h1_d_w44
                self.w45 -= learn_rate * d_L_d_ypred * d_ypred_d_h4 * d_h1_d_w45
                self.w46 -= learn_rate * d_L_d_ypred * d_ypred_d_h4 * d_h1_d_w46
                self.w47 -= learn_rate * d_L_d_ypred * d_ypred_d_h4 * d_h1_d_w47
                self.w48 -= learn_rate * d_L_d_ypred * d_ypred_d_h4 * d_h1_d_w48
                self.w49 -= learn_rate * d_L_d_ypred * d_ypred_d_h4 * d_h1_d_w49
                self.w410 -= learn_rate * d_L_d_ypred * d_ypred_d_h4 * d_h1_d_w410
                self.w411 -= learn_rate * d_L_d_ypred * d_ypred_d_h4 * d_h1_d_w411
                self.w412 -= learn_rate * d_L_d_ypred * d_ypred_d_h4 * d_h1_d_w412
                self.w413 -= learn_rate * d_L_d_ypred * d_ypred_d_h4 * d_h1_d_w413
                self.w414 -= learn_rate * d_L_d_ypred * d_ypred_d_h4 * d_h1_d_w414
                self.w415 -= learn_rate * d_L_d_ypred * d_ypred_d_h4 * d_h1_d_w415

                self.b4 -= learn_rate * d_L_d_ypred * d_ypred_d_h4 * d_h1_d_b4

                self.w1 -= learn_rate * d_L_d_ypred * d_ypred_d_w1
                self.w2 -= learn_rate * d_L_d_ypred * d_ypred_d_w2
                self.w3 -= learn_rate * d_L_d_ypred * d_ypred_d_w3
                self.w4 -= learn_rate * d_L_d_ypred * d_ypred_d_w4

                self.b5 -= learn_rate * d_L_d_ypred * d_ypred_d_b5

            if epoch % 30 == 0:
                y_preds = np.apply_along_axis(self.feedforward, 1, data)
                loss = mse_loss(all_y_trues, y_preds)
                print("Epoch %d loss: %.10f" % (epoch, loss))
                self.loss[self.sum] = loss

                self.sum = self.sum + 1
            if loss < 0.0001:
                break
# 文件的名字
FILENAME = "../testdata.xlsx"
# 禁用科学计数法
pd.set_option('float_format', lambda x: '%.3f' % x)
np.set_printoptions(suppress=True, threshold=sys.maxsize)
# 得到的DataFrame分别为点火温度	烧结机速	料层厚度	废气温度（南）	废气温度（北）	烟道负压（南）	烟道负压（北）	炉膛负压	焦炉煤气压力	焦炉煤气流量
# 空气压力	空气流量	排料温度	BTP温度（南）	BTP温度（北）	漏风率
data = pd.read_excel(FILENAME, header=0, usecols="A,B,C,D,E,F,G,H,I,J,K,L,M,N,O,P")
# DataFrame转化为array
DataArray = data.values
X = DataArray[:, 0:15]
Y = DataArray[:, 15]
X = np.array(X)#转化为array,自变量
Y = np.array(Y)#转化为array，因变量漏风率

# np.size()参数为1代表看有几列数据，参数为0代表看有几行数据
# np.sum()参数为1代表按行求和，参数为0代表按列求和
# 用每列数据的和除以每列数据的个数，即为该列属性的平均值
data = np.array(X)
data = MinMaxScaler().fit_transform(data)
# data_mean = np.sum(data, axis=0) / np.size(data, 0)
# data = data/data.max(axis=0)
# print(data)
all_y_trues = np.array(Y)
# all_y_trues_mean = np.sum(all_y_trues) / np.size(all_y_trues)
all_y_trues = all_y_trues / all_y_trues.max(axis=0)
# print(all_y_trues)
# 训练数据
network = OurNeuralNetwork()
network.train(data, all_y_trues)
# 输出神经网络参数
print("w11=%.5f" % network.w11)
print("w12=%.5f" % network.w12)
print("w13=%.5f" % network.w13)
print("w14=%.5f" % network.w14)
print("w15=%.5f" % network.w15)
print("w16=%.5f" % network.w16)
print("w17=%.5f" % network.w17)
print("w18=%.5f" % network.w18)
print("w19=%.5f" % network.w19)
print("w110=%.5f" % network.w110)
print("w111=%.5f" % network.w111)
print("w112=%.5f" % network.w112)
print("w113=%.5f" % network.w113)
print("w114=%.5f" % network.w114)
print("w115=%.5f" % network.w115)
print("w21=%.5f" % network.w21)
print("w22=%.5f" % network.w22)
print("w23=%.5f" % network.w23)
print("w24=%.5f" % network.w24)
print("w25=%.5f" % network.w25)
print("w26=%.5f" % network.w26)
print("w27=%.5f" % network.w27)
print("w28=%.5f" % network.w28)
print("w29=%.5f" % network.w29)
print("w210=%.5f" % network.w210)
print("w211=%.5f" % network.w211)
print("w212=%.5f" % network.w212)
print("w213=%.5f" % network.w213)
print("w214=%.5f" % network.w214)
print("w215=%.5f" % network.w215)
print("w31=%.5f" % network.w31)
print("w32=%.5f" % network.w32)
print("w33=%.5f" % network.w33)
print("w34=%.5f" % network.w34)
print("w35=%.5f" % network.w35)
print("w36=%.5f" % network.w36)
print("w37=%.5f" % network.w37)
print("w38=%.5f" % network.w38)
print("w39=%.5f" % network.w39)
print("w310=%.5f" % network.w310)
print("w311=%.5f" % network.w311)
print("w312=%.5f" % network.w312)
print("w313=%.5f" % network.w313)
print("w314=%.5f" % network.w314)
print("w315=%.5f" % network.w315)
print("w41=%.5f" % network.w41)
print("w42=%.5f" % network.w42)
print("w43=%.5f" % network.w43)
print("w44=%.5f" % network.w44)
print("w45=%.5f" % network.w45)
print("w46=%.5f" % network.w46)
print("w47=%.5f" % network.w47)
print("w48=%.5f" % network.w48)
print("w49=%.5f" % network.w49)
print("w410=%.5f" % network.w410)
print("w411=%.5f" % network.w411)
print("w412=%.5f" % network.w412)
print("w413=%.5f" % network.w413)
print("w414=%.5f" % network.w414)
print("w415=%.5f" % network.w415)
print("w1=%.5f" % network.w1)
print("w2=%.5f" % network.w2)
print("w3=%.5f" % network.w3)
print("w4=%.5f" % network.w4)
print("b1=%.5f" % network.b1)
print("b2=%.5f" % network.b2)
print("b3=%.5f" % network.b3)
print("b4=%.5f" % network.b4)
print("b5=%.5f" % network.b5)
# 标题显示中文
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False
# # 测试数据
# 1038.15	2.15	786.772	150.291	143.578	-15.098	-14.972	-2.986	4.304	1642.67	8.388	10337.5	23.976	387.627	406.398
testData = np.array([1038.15,2.15,786.772,150.291,143.578, -15.098,-14.972, -2.986,4.304,1642.67,8.388,10337.5,23.976,387.627,406.398])
testPrice = network.feedforward(testData)
# 损失函数曲线图
plt.plot(np.arange(100), network.loss)
plt.show()
# 真实值与预测值对比
y_preds = np.apply_along_axis(network.feedforward, 1, data)
plt.plot(np.arange(100), all_y_trues * 0.455586889,"r^")
plt.plot(np.arange(100),y_preds * 0.455586889,"bs")
plt.title("红色为真实值，蓝色为预测值")
plt.show()
print(testPrice)
print(y_preds * 0.455586889)

import numpy as np
import pandas as pd
import sys
from sklearn.preprocessing import MinMaxScaler

def sigmoid(x):
    # 第一层到第二层的激活函数
    return 1 / (1 + np.exp(-x))


class OurNeuralNetwork:
    def __init__(self):
        self.w11 = -0.42645
        self.w12 = 0.63565
        self.w13 = 1.35142
        self.w14 = 1.94237
        self.w15 = -0.29236
        self.w16 = 0.63735
        self.w17 = 2.02256
        self.w18 = 0.44977
        self.w19 = 0.08644
        self.w110 = -1.01199
        self.w111 = -0.22563
        self.w112 = 0.28637
        self.w113 = -0.85285
        self.w114 = -0.23621
        self.w115 = 0.02797
        self.w21 = 0.21159
        self.w22 = 0.20153
        self.w23 = 0.24574
        self.w24 = 2.17588
        self.w25 = -0.89182
        self.w26 = 0.54455
        self.w27 = 0.33510
        self.w28 = 0.46642
        self.w29 = -0.52934
        self.w210 = 1.53938
        self.w211 = -0.35739
        self.w212 = -1.04070
        self.w213 = 0.69904
        self.w214 = -0.25734
        self.w215 = 0.56322
        self.w31 = -1.79234
        self.w32 = -0.19899
        self.w33 = 0.21599
        self.w34 = -1.29977
        self.w35 = 1.30391
        self.w36 = -1.21500
        self.w37 = 0.28816
        self.w38 = -0.21522
        self.w39 = 0.25121
        self.w310 = -2.18489
        self.w311 = 0.58319
        self.w312 = 0.17534
        self.w313 = -0.61757
        self.w314 = -0.90510
        self.w315 = 1.59816
        self.w41 = 0.23268
        self.w42 = 1.11547
        self.w43 = -0.42850
        self.w44 = -2.07881
        self.w45 = 0.09449
        self.w46 = -0.85040
        self.w47 = -0.52319
        self.w48 = -0.63821
        self.w49 = -0.37806
        self.w410 = 0.88290
        self.w411 = -0.31316
        self.w412 = 0.66089
        self.w413 = 1.31396
        self.w414 = 0.54452
        self.w415 = -1.68133
        self.w1 = 0.20767
        self.w2 = 0.52343
        self.w3 = 0.48358
        self.w4 = 0.66367
        self.b1 = -0.30663
        self.b2 = 0.85381
        self.b3 = -0.96346
        self.b4 = -1.06187
        self.b5 = 0.21586

    def feedforward(self, x):
        # 前向传播学习
        h1 = sigmoid(
            self.w11 * x[0] + self.w12 * x[1] + self.w13 * x[2] + self.w14 * x[3] + self.w15 * x[4] + self.w16 * x[
                5] + self.w17 * x[6] + self.w18 * x[7] + self.w19 * x[8] + self.w110 * x[9] + self.w111 * x[
                10] + self.w112 * x[
                11] + self.w113 * x[12] + self.w114 * x[13] + self.w115 * x[14] + self.b1)
        h2 = sigmoid(
            self.w21 * x[0] + self.w22 * x[1] + self.w23 * x[2] + self.w24 * x[3] + self.w25 * x[4] + self.w26 * x[
                5] + self.w27 * x[6] + self.w28 * x[7] + self.w29 * x[8] + self.w210 * x[9] + self.w211 * x[
                10] + self.w212 * x[
                11] + self.w213 * x[12] + self.w214 * x[13] + self.w215 * x[14] + self.b2)
        h3 = sigmoid(
            self.w31 * x[0] + self.w32 * x[1] + self.w33 * x[2] + self.w34 * x[3] + self.w35 * x[4] + self.w36 * x[
                5] + self.w37 * x[6] + self.w38 * x[7] + self.w39 * x[8] + self.w310 * x[9] + self.w311 * x[
                10] + self.w312 * x[
                11] + self.w313 * x[12] + self.w314 * x[13] + self.w315 * x[14] + self.b3)
        h4 = sigmoid(
            self.w41 * x[0] + self.w42 * x[1] + self.w43 * x[2] + self.w44 * x[3] + self.w45 * x[4] + self.w46 * x[
                5] + self.w47 * x[6] + self.w48 * x[7] + self.w49 * x[8] + self.w410 * x[9] + self.w411 * x[
                10] + self.w412 * x[
                11] + self.w413 * x[12] + self.w414 * x[13] + self.w415 * x[14] + self.b4)

        o1 = self.w1 * h1 + self.w2 * h2 + self.w3 * h3 + self.w4 * h4 + self.b5
        return o1



# 文件的名字
FILENAME = "../测试集.xlsx"
# 禁用科学计数法
pd.set_option('float_format', lambda x: '%.3f' % x)
np.set_printoptions(suppress=True, threshold=sys.maxsize)
# 得到的DataFrame分别为点火温度	烧结机速	料层厚度	废气温度（南）	废气温度（北）	烟道负压（南）	烟道负压（北）	炉膛负压	焦炉煤气压力	焦炉煤气流量
# 空气压力	空气流量	排料温度	BTP温度（南）	BTP温度（北）	漏风率
data = pd.read_excel(FILENAME, header=0, usecols="A,B,C,D,E,F,G,H,I,J,K,L,M,N,O")
# DataFrame转化为array
DataArray = data.values
X = DataArray[:, 0:15]
X = np.array(X)#转化为array,自变量

network = OurNeuralNetwork()
data = np.array(X)
data = MinMaxScaler().fit_transform(data)

for i in data:
    testResult = network.feedforward(i)
    print(testResult * 0.45)

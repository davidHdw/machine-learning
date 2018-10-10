# -*- coding: utf-8 -*-
"""
Created on Tue Oct  9 18:32:46 2018

@author: 87955
"""

import numpy as np
import matplotlib.pyplot as plt

# 梯度上升算法：要找到某函数的最大值，最好的方法就是沿着该函数的梯度方向探寻。
# 梯度算子总是指向函数值増长最快的方向


def loadDataSet(filename):
    '''
    函数说明：加载数据集
    参数：需要加载的数据集的文件名
    返回：
        dataMat: 特征集
        labelMat: 标签集
    '''
    dataMat = []
    labelMat = []
    with open(filename) as fr:
        for line in fr.readlines():
            lineArr = line.strip().split()
            dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
            labelMat.append(int(lineArr[2]))
    return dataMat, labelMat

def sigmoid(z):
    '''
    函数说明：sigmoid函数
    '''
    return (1 / (1 + np.exp(-z)))

def logitModel(dataMat, ws):
    '''
    函数说明：logistics 模型
    参数：
        dataMat : 特征集
        ws : 回归参数(列向量)
    '''
    return sigmoid(np.dot(dataMat, ws))

def gradAscent(dataMatIn, classLabels):
    '''
    函数说明：梯度上升优化算法函数
    参数：
        dataMatIn： 特征集
        classLabels：标签集
    返回：
        weights : 回归系数
    '''
    dataMatrix = np.mat(dataMatIn)
    labelMat = np.mat(classLabels).transpose()
    m, n = np.shape(dataMatrix)
    alpha = 0.001
    maxCycle = 500
    weights = np.ones((n, 1))
    for k in range(maxCycle):
        h = logitModel(dataMatrix, weights)     # 预测结果
        # 对应博客中的第二种方法，只是还是求极大似然的值，没有转换成求最小值
        error = (labelMat - h)      # 误差
        weights = weights + alpha * dataMatrix.transpose() * error # (m,n) ===> (n,m) * (m,1) = (n,1)
    return weights

def plotBestFit(weights):
    '''
    函数说明：画出最佳决策边界
    参数：weights:回归系数
    '''
    dataMat, labelMat = loadDataSet("testSet.txt")
    dataArr = np.array(dataMat)
    m = np.shape(dataArr)[0]        # 样本个数
    xcord1 = []; ycord1 = []
    xcord2 = []; ycord2 = []
    for i in range(m):
        if int(labelMat[i] == 1):
            xcord1.append(dataArr[i, 1])
            ycord1.append(dataArr[i, 2])
        else :
            xcord2.append(dataArr[i, 1])
            ycord2.append(dataArr[i, 2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker="s")
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x = np.matrix(np.arange(-3.0, 3.0, 0.1))
    y = (-weights[0] - weights[1] * x) / weights[2]
    print("x.shape: ",x.shape, "  type(x): ", type(x))
    print("y.shape: ",y.shape, "  type(y): ", type(y))
    ax.plot(x, y)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()
    
def stocGradAscent0(dataMatrix, classLabels):
    '''
    函数说明：随机梯度上升优化算法
    参数：dataMatrix：特征数据集
          classLabels： 标签集
    返回：回归系数
    '''
    m,n = np.shape(dataMatrix)
    alpha = 0.01
    weights = np.ones(n)
    for i in range(m):
        h = logitModel(dataMatrix[i], weights)
        error = classLabels[i] - h
        weights = weights + alpha * error * dataMatrix[i]
    return weights

def stocGradAscent1(dataMatrix, classLabels, numIt=50):
    '''
    函数说明：改进版的随机梯度上升优化算法，采用样本随机选择和alpha动态减少机制，比固定alpha的方法收敛更快
    参数：
        dataMatrix：特征集
        classLabels：标签集
        numIt ：迭代次数，默认为50
    返回：
        weights : 回归系数
    '''
    m, n = np.shape(dataMatrix)
    weights = np.ones(n)
    for j in range(numIt):
        dataIndex = list(range(m))
        for i in range(m):
            alpha = 4 / (1.0 + j + i)+0.01
            randIndex = int(np.random.uniform(0, len(dataIndex)))
            h = logitModel(dataMatrix[randIndex], weights)
            error = classLabels[randIndex] - h
            weights = weights + alpha * dataMatrix[randIndex] * error
            del(dataIndex[randIndex])
    return weights
    


if __name__ == "__main__":
    dataMat, labelMat = loadDataSet("testSet.txt")
    weights = gradAscent(dataMat, labelMat)
    print(weights)
    plotBestFit(weights.getA())
    weights = stocGradAscent1(np.array(dataMat), labelMat)
    plotBestFit(weights)
        
    
    
    
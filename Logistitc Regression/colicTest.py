# -*- coding: utf-8 -*-
"""
Created on Wed Oct 10 14:37:47 2018

@author: 87955
"""

##########################################
# 示例：从疝气病症预测病马的死亡率
#########################################

'''
处理数据中的缺失值的一些可行的做法：
1、使用可用特征的均值来填补缺失值
2、使用特征值来填补缺失值，如-1
3、忽略有缺失值的样本
4、使用相似样本的均值添补缺失值
5、实用另外的机器学习算法预测缺失值
'''

# 第一步：数据预处理
# 1、缺失值用0替换
# 2、若标签已经缺失，则直接舍弃该数据集


import numpy as np


def sigmoid(z):
    return (1.0/ (1.0+np.exp(-z)))

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
            h = logisticModel(dataMatrix[randIndex], weights)
            error = classLabels[randIndex] - h
            weights = weights + alpha * dataMatrix[randIndex] * error
            del(dataIndex[randIndex])
    return weights

def logisticModel(X, weights):
    return sigmoid(np.dot(X, weights))

def classifyVector(X, weights):
    '''
    函数说明：分类函数
    参数：
        X：特征集
        weights：回归系数（列向量）
    返回：分类结果
    '''
    prob = logisticModel(X, weights)
    if prob > 0.5:
        return 1
    return 0

def colicTest():
    '''
    
    '''
    frTrain = open('horseColicTraining.txt')
    frTest = open("horseColicTest.txt")
    trainingSet = []
    trainingLabels = []
    for line in frTrain.readlines():
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[21]))
    trainWeights = stocGradAscent1(np.array(trainingSet), trainingLabels, 500)
    errorCount = 0
    numTestVec = 0.0
    for line in frTest.readlines():
        numTestVec += 1.0
        currLine = line.strip().split('\t')
        linnArr = []
        for i in range(21):
            linnArr.append(float(currLine[i]))
        if int(classifyVector(np.array(lineArr), trainWeights)) != int(currLine[21]) : 
            errorCount += 1
    errorRate = (float(errorCount)/numTestVec)
    print("the error rate of this test is: %f"%errorRate)
    return errorRate

def multiTest():
    numTest = 10
    errorSum = 0.0
    for k in range(numTest):
        errorSum += colicTest()
    print("after %d iterations the average error rate is:%f" % (numTest, errorSum/float(numTest)))


if __name__ == "__main__":
    multiTest()
    
    
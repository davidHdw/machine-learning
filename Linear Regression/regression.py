# -*- coding: utf-8 -*-
"""
Created on Fri Oct  5 10:47:15 2018

@author: david
"""

import numpy as np
import matplotlib.pyplot as plt

def loadDataSet(fileName):
    '''
    函数说明：数据导入函数
    参数：fileName: 数据存放的路径
    返回：dataMat: 数据特征集 
         labelMat: 数据标签集
    '''
    dataMat = []
    labelMat = []
    with open(fileName) as fr:
        numFeat = len(fr.readline().split("\t")) - 1
        for line in fr.readlines():
            lineArr = []
            curLine = line.strip().split('\t')
            for i in range(numFeat):
                lineArr.append(float(curLine[i]))
            dataMat.append(lineArr)
            labelMat.append(float(curLine[-1]))
    return dataMat, labelMat

def standRegres(xArr, yArr):
    '''
    函数说明：标准回归函数（使用标准最小二乘法）
    参数: xArr: 不包括标签的训练集Array数组
         yArr: 训练集标签的Array数组
    返回：ws: 参数w的向量
    '''
    xMat = np.mat(xArr)
    yMat = np.mat(yArr).T       #转化成列向量
    # 按公式进行参数w的求解
    xTx = xMat.T * xMat
    if np.linalg.det(xTx) == 0.0:   # 行列式是否为0，是不能进行求逆
        print('This matrix is singular, cannot do inverse')
        return 
    ws = xTx.I * (xMat.T * yMat)
    print("ws:", ws)
    return ws
    
def plotDataSet():
    '''
    函数说明：绘制数据集
    参数：无
    返回：无
    '''
    xArr, yArr = loadDataSet("ex0.txt")     # 加载数据集
    n = len(xArr)                           # 获取数据长度
    xcord = []                              # 样本点
    ycord = []
    for i in range(n):
        xcord.append(xArr[i][1])            # xArr每行有一个偏置b和一个特征x
        ycord.append(yArr[i])               # yArr每行只有对应的值
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord, ycord, s=20, c='blue', alpha=0.5) # 描绘样本
    plt.title('DataSet')
    plt.xlabel("X")
    plt.show()

def plotRegression():
    '''
    函数说明：绘制回归曲线和数据点
    参数：无
    返回：无
    '''
    xArr, yArr = loadDataSet("ex0.txt")
    ws = standRegres(xArr, yArr)        # 调用标准回归函数获取回归参数ws
    xMat = np.mat(xArr)                 # 把array转成matrix
    yMat = np.mat(yArr)
    xCopy = xMat.copy()                 # 深拷贝xMat,得到xCopy，使的对xCopy操作时不影响xMat
    xCopy.sort(0)                       # 对数据集进行排序，从小到大
    yHat = xCopy * ws                   # 计算对应的回归y
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(xCopy[:,1], yHat, c = 'red')    # 画回归线
    # 描绘数据集个点，关于yMat.flatten.A[0]的意思：https://blog.csdn.net/lilong117194/article/details/78288795
    ax.scatter(xMat[:,1].flatten().A[0], yMat.flatten().A[0],
               s = 20, c = 'blue', alpha = .5)       
    plt.title("DataSet and Regression")
    plt.xlabel("X")
    plt.show()
    
    
if __name__ == "__main__":
    plotDataSet()
    plotRegression()
    
    
    
    
    
    
    
    
    
    
    
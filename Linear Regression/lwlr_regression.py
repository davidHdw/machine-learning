# -*- coding: utf-8 -*-

# lwlr_regression.py

"""
Created on Fri Oct  5 15:59:57 2018

@author: 87955
"""

import numpy as np
import matplotlib.pyplot as plt

def loadDataSet(fileName):
    '''
    函数说明：数据导入函数
    参数：fileName: 数据存放的路径
    返回：dataMat:  数据特征集 
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

def lwlr(testPoint, xArr, yArr, k=1.0):
    '''
    函数说明：利用局部加权回归求解回归系数w,并且进行预测
    参数：
        testPoint : 测试样本点
        xArr :      x数据集
        yArr :      y数据集
        k :         高斯核的k,默认k=1.0
    返回：
        ws * testPoint : 测试样板点的测试结果        
    '''
    xMat = np.mat(xArr)
    yMat = np.mat(yArr).T       # 转换类型成matrix,并对其进行转置
    m = np.shape(xMat)[0]       # 样本数
    weights = np.mat(np.eye((m)))   #创建对角矩阵
    for i in range(m):
        diffMat = testPoint - xMat[i,:] # 当距离越远，则值越大，特征即坐标。
        weights[i,i] = np.exp(diffMat*diffMat.T / (-2.0*k**2)) # 成指数下降
    xTwx = xMat.T * (weights * xMat)                            ###############
    if np.linalg.det(xTwx) == 0:        #判断是否为0矩阵         # 按公式
        print("This matrix is singular, cannot do inverse")     # 进行参数
        return                                                  # w的求解
    ws = xTwx.I * (xMat.T * (weights * yMat))                   ###############
    return testPoint * ws

def lwlrTest(testArr, xArr, yArr, k=1.0):
    '''
    函数说明：局部加权回归测试
    参数：
        testArr: 测试集
        xArr:    x数据集
        yArr:    y数据集
        k:       高斯核中的k,默认为1.0
    返回：
        yHat:    测试结果集
    '''
    m = np.shape(testArr)[0]   #测试样本个数
    yHat = np.zeros(m)
    for i in range(m):                              # 对每个样本进行预测
        yHat[i] = lwlr(testArr[i], xArr, yArr, k)   
    return yHat
    
def plotLwlrRegression():
    '''
    函数说明：绘制多条k不同的局部加权线性回归曲线
    参数：无
    返回：无
    '''
    xArr, yArr = loadDataSet("ex0.txt")
    yHat1 = lwlrTest(xArr, xArr, yArr, 1.0)         # 设置不同的k值进行预测
    yHat2 = lwlrTest(xArr, xArr, yArr, 0.01)
    yHat3 = lwlrTest(xArr, xArr, yArr, 0.003)
    xMat = np.mat(xArr)                             # 转化成matrix类型
    yMat = np.mat(yArr)
    sortInd = xMat[:,1].argsort(0)
    xSort = xMat[sortInd][:,0,:]        # 按照升序进行排序
    fig, axs = plt.subplots(nrows=3, ncols=1, sharex=False,
                            sharey=False, figsize=(10,8))
    
    axs[0].plot(xSort[:,1], yHat1[sortInd], c='red')    # 分别画出不同k值的回归曲线
    axs[1].plot(xSort[:,1], yHat2[sortInd], c='red')
    axs[2].plot(xSort[:,1], yHat3[sortInd], c='red')
    
    # 在3个子图中描绘样本点
    axs[0].scatter(xMat[:, 1].flatten().A[0], yMat.flatten().A[0],
       s = 20, c = 'blue', alpha = .5)
    axs[1].scatter(xMat[:, 1].flatten().A[0], yMat.flatten().A[0],
       s = 20, c = 'blue', alpha = .5)
    axs[2].scatter(xMat[:, 1].flatten().A[0], yMat.flatten().A[0],
       s = 20, c = 'blue', alpha = .5)
    
    # 设置标题，x轴label，y轴label
    axs0_title_text = axs[0].set_title('lwlr k=1.0')
    axs1_title_text = axs[1].set_title('lwlr k=0.01')
    axs2_title_text = axs[2].set_title('lwlr k=0.003')
    
    plt.setp(axs0_title_text, size=8, weight='bold',color='red')
    plt.setp(axs1_title_text, size=8, weight='bold',color='red')
    plt.setp(axs2_title_text, size=8, weight='bold',color='red')
    
    plt.xlabel("X")
    plt.show()
    

if __name__ == "__main__":
    plotLwlrRegression()
    
    
    
    
    
    
    
    
    
    
    
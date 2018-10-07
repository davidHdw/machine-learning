# -*- coding: utf-8 -*-

# ridgeRegression.py

import numpy as np
import matplotlib.pyplot as plt

def rssError(yArr, yHatArr):
    """
    函数说明：求平方误差
    """
    return ((yArr-yHatArr)**2).sum()

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

def ridgeRegre(xMat, yMat, lam=0.2):
    '''
    函数说明：岭回归
    参数：
        xMat: x数据集
        yMat: y数据集
        lam:  缩减系数
    返回：
        ws : 岭回归的回归系数
    '''
    xTx = xMat.T * xMat                             # 按公式进行求解系数
    denom = xTx + np.eye(np.shape(xMat)[1]) * lam
    if np.linalg.det(denom) == 0.0:
        print("This matrix is singular, cannot do inverse")
        return 
    ws = denom * (xMat.T * yMat)
    return ws

def ridgeTest(xArr, yArr):
    '''
    函数说明：岭回归测试
    参数：
        xArr: x数据集
        yArr: y数据集
    返回：
        wMat : 回归系数
    '''
    xMat = np.mat(xArr)
    yMat = np.mat(yArr).T
    yMean = np.mean(yMat, axis = 0)    # axis=0,表示纵轴(列)，1代表横轴(行)，此处对列进行求平均
    yMat = yMat - yMean
    xMeans = np.mean(xMat, axis = 0)
    xVar = np.var(xMat, axis = 0)      # 求方差
    xMat = (xMat - xMeans) / xVar       # 进行归一化，使得变成标准化
    numTestPst = 30                     # 30个不同的lam进行测试
    wMat = np.zeros((numTestPst, np.shape(xMat)[1]))    # 初始化系数矩阵
    for i in range(numTestPst):
        ws = ridgeRegre(xMat, yMat, lam=np.exp(i-10))   # 对不同的lam值进行求解回归系数，lam以指数级变化，从很小的数开始
        wMat[i, :] = ws.T               # 对每个lam得到的回归系数进行保存
    return wMat

def plotwMat():
    '''
    函数说明：绘制岭回归系数矩阵
    参数：无
    返回：无
    '''
    abX, abY = loadDataSet("abalone.txt")
    ridgeWeights = ridgeTest(abX, abY)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(ridgeWeights)
    ax_title_text = ax.set_title('gaun xi')
    ax_xlabel_text = ax.set_xlabel('log(lambada)')
    ax_ylabel_text = ax.set_ylabel('ws')    
    plt.setp(ax_title_text, size=20, weight="bold", color = 'red')
    plt.setp(ax_xlabel_text, size=10, weight="bold", color = 'black')
    plt.setp(ax_ylabel_text, size=10, weight="bold", color = 'black')
    plt.show()
    
if __name__ == "__main__":
    plotwMat()








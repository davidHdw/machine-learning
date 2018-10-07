# -*- coding: utf-8 -*-

# stageWiseRegression.py

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

def rssError(yArr, yHatArr):
    """
    函数说明：求平方误差
    """
    return ((yArr-yHatArr)**2).sum()

def regularize(xMat, yMat):
    '''
    函数说明：数据标准化，标准化：均值为0，方差为1
    参数：
        xMat : x数据集
        yMat : y数据集
    返回：
        inxMat : 标准化后的x数据集
        inyMat : 标准化后的y数据集
    '''
    inxMat = xMat.copy()            # 数据拷贝
    inyMat = yMat.copy()
    yMean = np.mean(inyMat, axis=0)     # 对列进行求均值
    inyMat = yMat - yMean               # 减去均值
    inMeans = np.mean(inxMat, axis=0)   # 对列进行求均值
    inVar = np.var(inxMat, axis=0)      # 对列进行求方差
    inxMat = (inxMat - inMeans) / inVar # 除以总方差，实现标准化
    return inxMat, inyMat
    
def stageWise(xArr, yArr, eps=0.01, numIt = 100):
    '''
    函数说明：前向逐步线性回归
    参数：
        xArr:  x输入数据
        yArr:  y预测数据
        eps:   每次调整的步长
        numIt: 迭代次数
    返回：
        returnMat:
    '''
    xMat = np.mat(xArr)
    yMat = np.mat(yArr).T
    xMat, yMat = regularize(xMat, yMat)     # 对数据进行标准化
    m,n = np.shape(xMat)
    returnMat = np.zeros((numIt, n))
    ws = np.zeros((n,1));
    wsTest = ws.copy()
    wsMax = ws.copy()
    for i in range(numIt):          # 迭代numIt次
        print(ws.T)
        lowestError = np.inf        # 每次迭代领最小误差为正无穷
        for j in range(n):          # 对每个特征
            for sign in [-1, 1]:    # 减或加   
                wsTest = ws.copy()
                wsTest[j] += eps * sign
                yTest = xMat * wsTest
                rssE = rssError(yMat.A, yTest.A) # 求总误差
                if rssE < lowestError :
                    lowestError = rssE
                    wsMax = wsTest
        ws = wsMax.copy()
        returnMat[i, :] = ws.T      #记录numIt次迭代的回归系数矩阵
    return returnMat
    
def plotstageWiseMat():
	"""
	函数说明:绘制岭回归系数矩阵

	"""
	xArr, yArr = loadDataSet('abalone.txt')
	returnMat = stageWise(xArr, yArr, 0.005, 1000)
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.plot(returnMat)	
	ax_title_text = ax.set_title('die dai ci shu yu hui gui xi shu de guan xi ')
	ax_xlabel_text = ax.set_xlabel('diedai cishu')
	ax_ylabel_text = ax.set_ylabel('huigui xishu')
	plt.setp(ax_title_text, size = 15, weight = 'bold', color = 'red')
	plt.setp(ax_xlabel_text, size = 10, weight = 'bold', color = 'black')
	plt.setp(ax_ylabel_text, size = 10, weight = 'bold', color = 'black')
	plt.show()


if __name__ == '__main__':
	plotstageWiseMat()    
    

    
    
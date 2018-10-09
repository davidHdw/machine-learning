# -*- coding: utf-8 -*-
"""
Created on Mon Oct  8 17:25:33 2018

@author: 87955
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def loadDataSet(filename):
    '''
    函数说明：读取数据
    参数：
        filename：文件名
    返回：
        pdData : 数据集
    '''
    pdData = pd.read_csv(filename, header=None, names=['Exam 1','Exam 2','Admitted'])
    return pdData

def plotDataSet(pdData):
    '''
    函数说明：对数据集进行可视化
    参数：pdData : DataFrame类型数据集
    '''
    positive = pdData[pdData['Admitted'] == 1]  # 正例
    negative = pdData[pdData['Admitted'] == 0]  # 反例
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(positive['Exam 1'], positive['Exam 2'], s=30, c='b', marker='o', label='Admitted')
    ax.scatter(negative['Exam 1'], negative['Exam 2'], s=30, c='r', marker='x', label='No Admitted')
    ax.legend()
    ax.set_xlabel('Exam 1 Score')
    ax.set_ylabel("Exam 2 Score")
    
    
def sigmoid(z):
    '''
    函数说明：sigmoid函数
    '''
    return (1 / (1+np.exp(-z)))

def logisticModel(X, ws):
    '''
    函数说明：逻辑斯蒂回归模型
    参数：
        X:  数据集（不包括标签）
        ws: 回归参数
    返回：
        当前预测结果
    '''
    return sigmoid(np.dot(X, ws.T))
    
def cost(X, y, ws):
    '''
    函数说明：求平均损失函数
    参数：
        X: 数据集（不含标签）
        y: 标签
    返回：
        平均损失
    '''
    left = np.multiply(-y, np.log(logisticModel(X, ws)))
    right = np.multiply(1-y, np.log(1 - logisticModel(X, ws)))
    return np.sum(left - right) / (len(X))

def gradient(X, y, ws):
    '''
    函数说明：梯度函数
    参数：
        X: 数据集（不含标签）
        y: 标签        
    返回：
        梯度值
    '''
    grad = np.zeros(ws.shape)   
    error = (logisticModel(X, ws) - y).ravel() # 误差
    for j in range(len(ws.ravel())):
        term = np.multiply(error, X[:,j])
        grad[0, j] = np.sum(term) / len(X)
        
    return grad

STOP_ITER = 0
STOP_COST = 1
STOP_GRAD = 2

def stopCriterion(tp, value, threshold):
    '''
    设置三种不同的停止策略
    '''
    if tp == STOP_ITER:
        return value > threshold
    elif tp == STOP_COST:
        return abs(value[-1]-value[-2]) < threshold
    elif tp == STOP_GRAD:
        return np.linalg.norm(value) < threshold
    
def shuffleData(data):
    '''
    函数说明：对数据进行洗牌
    '''
    np.random.shuffle(data)
    cols = data.shape[1]
    X = data[:, 0:cols-1]
    y = data[:, cols-1:]
    return X,y

import time

def graDescent(data, ws, batchSize, stopType, thresh, alpha):
    '''
    函数说明：梯度下降求解
    参数：
        data：数据集
        ws：回归参数
        batchSize：大小
        stopType：停止更新模式标志
        thresh：阀值
        alpha： 学习率
    返回：
    '''
    init_time = time.time()
    i = 0  # 迭代次数
    k = 0  # batch
    X, y = shuffleData(data)
    m = X.shape[0]           # 样本个数
    grad = np.zeros(ws.shape)  # 初始化梯度
    costs = [cost(X, y, ws)]   # 损失值
    
    while True:
        grad = gradient(X[k:k+batchSize], y[k:k+batchSize], ws)
        k += batchSize
        if k >= m:
            k = 0
            X, y = shuffleData(data)  # 重新洗牌
        ws = ws - alpha * grad        # 参数更新
        costs.append(cost(X,y, ws))   # 计算新的损失
        i += 1
        
        if stopType == STOP_ITER: 
            value = i
        elif stopType == STOP_COST:
            value = costs
        elif stopType == STOP_GRAD:
            value = grad
        
        if stopCriterion(stopType, value, thresh):
            break
        
    return ws, i-1, costs, grad, time.time() - init_time

def runExpe(data, ws, batchSize, stopType, thresh, alpha):
    ws, iter, costs, grad, dur = graDescent(data, ws, batchSize, stopType, thresh, alpha)
    name = "Original" if (data[:,1]>2).sum() > 1 else "Scaled"
    name += " data - learning rate: {} - ".format(alpha)
    if batchSize == data.shape[0]:
        strDescType = "Gradient"
    elif batchSize == 1:
        strDescType = "Stochastic"
    else: strDescType = "Mini-batch({})".format(thresh)
    
    name += strDescType + " descent - Stop: "
    if stopType == STOP_ITER:
        strStop = "{} iterations".format(thresh)
    elif stopType == STOP_COST:
        strStop = "costs change < {}".format(thresh)
    else: strStop = "gradient norm < {}".format(thresh)
    
    name += strStop
    print("***{}\nTheta: {} - Iter: {} - Last cost: {:03.2f} - Duration: {:03.2f}"
          .format(name, ws, iter, costs[-1], dur))
    fig, ax = plt.subplots(figsize=(12,4))
    ax.plot(np.arange(len(costs)), costs, 'r')
    ax.set_xlabel("Iterations")
    ax.set_ylabel("Cost")
    ax.set_title(name.upper()+" - Error vs. Iteration")
    return ws

#设定阈值
def predict(X, theta):
    return [1 if x >= 0.5 else 0 for x in logisticModel(X, theta)]



if __name__ == "__main__":
    pdData = loadDataSet('LogiReg_data.txt')
    plotDataSet(pdData)
    pdData.insert(0, "ones", 1) # 插入x0 = 1
    orig_data = pdData.as_matrix() # 把DataFrame转成matrix类型
    cols = orig_data.shape[1]
    X = orig_data[:,0:cols-1]
    y = orig_data[:,cols-1:]
    theta = np.zeros([1,3])
    
    runExpe(orig_data, theta, 100, STOP_ITER, thresh=5000, alpha=0.000001)
    runExpe(orig_data, theta, 100, STOP_COST, thresh=0.000001, alpha=0.001)
    runExpe(orig_data, theta, 100, STOP_GRAD, thresh=0.05, alpha=0.001)
    runExpe(orig_data, theta, 1, STOP_ITER, thresh=5000, alpha=0.001)
    runExpe(orig_data, theta, 16, STOP_ITER, thresh=15000, alpha=0.001)
    
    # 对数据进行预处理：标准化（均值为0，方差为1）
    import sklearn.preprocessing as pp
    scaled_data = orig_data.copy()
    scaled_data[:,1:3] = pp.scale(scaled_data[:, 1:3])
    print(scaled_data[:5])
    
    runExpe(scaled_data, theta, 100, STOP_GRAD, thresh=0.02, alpha=0.001)
    theta = runExpe(scaled_data, theta, 1, STOP_GRAD, thresh=0.002/5, alpha=0.001)
    runExpe(scaled_data, theta, 16, STOP_GRAD, thresh=0.002*2, alpha=0.001)
    
    # 正确率验证
    scaled_X = scaled_data[:, :3]
    y = scaled_data[:, 3]
    predictions = predict(scaled_X, theta)
    correct = [1 if ((a == 1 and b == 1) or (a == 0 and b == 0)) else 0 for (a, b) in zip(predictions, y)]
    accuracy = (sum(map(int, correct)) % len(correct))
    print ('accuracy = {0}%'.format(accuracy))
    
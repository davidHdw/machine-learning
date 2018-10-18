# -*- coding: utf-8 -*-

# Bayes.py

import numpy as np

def loadDataSet():
    postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help','please'],
                   ['maybe', 'not', 'take', 'him', 'to','dog','park','stupid'],
                   ['my', 'dalmation', 'is','so','cute', 'I', 'love','him'],
                   ['stop', 'posting', 'stupid','worthless', 'garbage'],
                   ['mr', 'licks','ate','my','steak', 'how', 'to','stop','him'],
                   ['quit','buying','worthless','dog','food','stupid']]
    classVec = [0,1,0,1,0,1]
    return postingList, classVec
    
def createVocabList(dataSet):
    '''
    函数说明：将实验样本处理成词汇表
    输入：
        dataSet : 整理的样本数据集
    返回：
        vocabSet : 词汇表
    '''
    vocabSet = set([])
    for document in dataSet:
        vocabSet = vocabSet | set(document)
    return list(vocabSet)

def setOfWords2Vec(vocabList, inputSet):
    '''
    函数说明：根据vocabList词汇表，将inputSet向量化，向量的元素为1或0
    词集模式：每个词的出现与否作为特征，即出现则为1不管出现的次数，不出现则为0
    输入：
        vocabList：createVocabList返回的词汇表
        inputSet：切分的词条列表
    返回：
        returnVec : 文档向量，词集模型
    '''
    returnVec = [0] * len(vocabList)    # 创建一个所有元素都为0的向量
    for word in inputSet:               # 遍历每个词条
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1    # 词存在词汇表中，则置1
        else :
            print("The word: %s is not in my Vocabulary!" % word)
    return returnVec

def bagOfWord2VecMN(vocabList, inputSet):
    '''
    函数说明：根据vocabList词汇表，将inputSet向量化，向量的元素对应词出现的次数
    词袋模式：
    输入：
        vocabList：createVocabList返回的词汇表
        inputSet：切分的词条列表
    返回：
        returnVec : 文档向量，词集模型
    '''
    returnVec = [0] * len(vocabList)    # 创建一个所有元素都为0的向量
    for word in inputSet:               # 遍历每个词条
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1    # 词存在词汇表中，则出现次数自加1
    return returnVec    

def trainNB0(trainMatrix, trainCategory):
    '''
    函数说明：朴素贝叶斯分类器训练函数
    输入：
        trainMatrix：训练文档矩阵，即setOfWords2Vec返回的returnVec构成的矩阵
        trainCategory：训练类别标签向量，即loadDataSet返回的classVec
    返回：
        p0Vect： 非的条件概率数组, P(x_i|c=0) 
        p1Vect： 侮辱类的条件概率数组, P(x_i|c=1)
        pAbusive：文档属于侮辱类的概率，即P(C=1),而P(C=0)=1-P(c=1)
    '''
    numTrainDocs = len(trainMatrix) # 文章数目
    numWords = len(trainMatrix[0])  # 文章长度，因经过setOfWords2Vec处理，所以都一样长
    pAbusive = sum(trainCategory) / float(numTrainDocs) # 类别为1的先验概率P(c=1)
    # 创建一个长度为numWords且值全为0的数组p0Num，
    # 用于统计在类别为0的训练样本中各个属性(此处为词)的出现的次数
    p0Num = np.zeros(numWords) 
    p1Num = np.zeros(numWords)
    p0Demo = 0.0
    p1Demo = 0.0
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            # 出现的词自加1
            p1Num += trainMatrix[i]
            # 该类别的总词数加上当前样本的词数
            p1Demo += sum(trainMatrix[i])
        else :
            p0Num += trainMatrix[i]
            p0Demo += sum(trainMatrix[i])
    p1Vect = p1Num/p1Demo # 每个元素除以该类别中的总词数
    p0Vect = p0Num/p0Demo
    return p0Vect, p1Vect, pAbusive
    
def trainNB1(trainMatrix, trainCategory):
    '''
    函数说明：朴素贝叶斯分类器训练函数,为防止出现概率为0导致，分类出现偏差，采用“拉普拉斯平滑”
             又为防止连乘出现数值下溢，对结果进行取对数
    输入：
        trainMatrix：训练文档矩阵，即setOfWords2Vec返回的returnVec构成的矩阵
        trainCategory：训练类别标签向量，即loadDataSet返回的classVec
    返回：
        p0Vect： 非的条件概率数组, P(x_i|c=0) 
        p1Vect： 侮辱类的条件概率数组, P(x_i|c=1)
        pAbusive：文档属于侮辱类的概率，即P(C=1),而P(C=0)=1-P(c=1)
    '''
    numTrainDocs = len(trainMatrix) # 文章数目
    numWords = len(trainMatrix[0])  # 文章长度，因经过setOfWords2Vec处理，所以都一样长
    pAbusive = (sum(trainCategory)+1) / float(numTrainDocs+2) # 类别为1的先验概率P(c=1)
    # 创建一个长度为numWords且值全为1的数组p0Num，(参考拉普拉斯修正)
    # 用于统计在类别为0的训练样本中各个属性(此处为词)的出现的次数
    p0Num = np.ones(numWords) 
    p1Num = np.ones(numWords)
    p0Demo = 2.0  # 每个词(属性)只会出现为0,1;所以初始为2.0
    p1Demo = 2.0
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            # 出现的词自加1
            p1Num += trainMatrix[i]
            # 该类别的总词数加上当前样本的词数
            p1Demo += sum(trainMatrix[i])
        else :
            p0Num += trainMatrix[i]
            p0Demo += sum(trainMatrix[i])
    p1Vect = p1Num/p1Demo # 每个元素除以该类别中的总词数
    p0Vect = p0Num/p0Demo
    return np.log(p0Vect), np.log(p1Vect), pAbusive

def classifyNB(vec2Classify, p0Vect, p1Vect, pClass1):
    '''
    函数说明：朴素贝叶斯分类器分类函数
    参数：
        vec2Classify：待分类的词条数组
        p0Vect：非侮辱类的条件概率数组
        p1Vect：侮辱类的条件概率数组
        pClass1 ： 侮辱类的先验概率
    返回：
        1 : 侮辱类
        0 ：非侮辱类
    '''
    p1 = sum(p1Vect * vec2Classify) + np.log(pClass1)
    p0 = sum(p0Vect * vec2Classify) + np.log(1 - pClass1)
    if p1 > p0 :
        return 1
    else:
        return 0
    
def testingNB():
    '''
    函数说明:测试朴素贝叶斯分类器
    '''
    listOPosts, listClass = loadDataSet()
    myVocabList = createVocabList(listOPosts)
    print(myVocabList)
    print(setOfWords2Vec(myVocabList, listOPosts[0]))
    trainMat = []
    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
    p0v, p1v, pAb = trainNB1(trainMat, listClass)    
    testEntry = ['love', 'my','dalmation']
    thisDoc = np.array(setOfWords2Vec(myVocabList, testEntry))
    print(testEntry, 'classifyed as: ','侮辱类' if classifyNB(thisDoc, p0v, p1v, pAb) else '非侮辱类')
    testEntry = ['stupid', 'garbage']
    thisDoc = np.array(setOfWords2Vec(myVocabList, testEntry))
    print(testEntry, 'classifyed as: ','侮辱类' if classifyNB(thisDoc, p0v, p1v, pAb) else '非侮辱类')    
    
if __name__ == "__main__":
    testingNB()
    

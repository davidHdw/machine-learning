# -*- coding: utf-8 -*-

# Bayes-spam.py

import numpy as np
import Bayes

def textParse(bigString):
    '''
    函数说明：
    
    '''
    import re
    listOpTokens = re.split(r'\W*', bigString)
    return [tok.lower() for tok in listOpTokens if len(tok) > 2]

def spamTest():
    '''
    '''
    docList = []
    classList = []
    fullText = []
    for i in range(1, 26):
        wordList = textParse(open('email/spam/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        wordList = textParse(open('email/ham/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = Bayes.createVocabList(docList)
    trainingSet = list(range(50))
    testSet = []
    for i in range(10):
        randIndex = int(np.random.uniform(0, len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])
    trainMat = []
    trainClasses = []
    for docIndex in trainingSet:
        trainMat.append(Bayes.setOfWords2Vec(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0v, p1v, pSpam = Bayes.trainNB1(np.array(trainMat), np.array(trainClasses))
    errorCount = 0
    for docIndex in testSet:
        wordVector = Bayes.setOfWords2Vec(vocabList, docList[docIndex])
        if Bayes.classifyNB(np.array(wordVector), p0v, p1v, pSpam) \
        != classList[docIndex]:
            errorCount += 1
    print("the error rate is: ", float(errorCount/len(testSet)))
    
if __name__ == '__main__':
    spamTest()
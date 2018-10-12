# -*- coding: utf-8 -*-
"""
Created on Thu Oct 11 15:33:49 2018

@author: 87955
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import KFold
from sklearn.metrics import confusion_matrix, recall_score
import itertools
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier

def loadDataSet(filename):
    '''
    函数说明：读取数据
    参数：
        filename ：数据文件名
    返回：data ：数据集
    '''
    data = pd.read_csv(filename)
    print(data.head())          # 显示前5个样本
    return data

def plotDataSet(data):
    '''
    函数说明：图形显示正反样本的个数
    参数：data：数据集
    '''
    # 对class列的值的个数进行统计，即此处为统计1和0的各自出现的次数
    count_classes = pd.value_counts(data["Class"], sort=True).sort_index()
    count_classes.plot(kind = "bar") # 使用pd自带的画图函数数据进行可视化，采用条形图
    plt.title("Fraud class histogram")
    plt.xlabel("Class")
    plt.ylabel("frequency")
    print(count_classes)
    
def dataPreProcess(data):
    '''
    函数说明：数据预处理
    参数：data:原数据集
    返回：
        data ：预处理之后的整个数据集
        X ：特征集
        y : 标签集
        
    注：一般来说，我们的数据集中会存在缺失值，还应进行缺失值处理，
        但该数据集已经是进行过处理的数据集，没有缺失值，所以不用进行缺失值处理
    '''
    # 对data中特征Amount的值进行标准化处理，均值为0，方差为1，并添加到data中
    data["normalAmount"] = StandardScaler().fit_transform(data["Amount"].values.reshape(-1, 1))
    # axis=1,表示对列进行操作，即drop(丢掉)Time和Amount两个特征
    data = data.drop(["Time", "Amount"], axis=1) 
    print(data.head()) # 查看是否标准化和drop成功
    X = data.iloc[:, data.columns != "Class"]        # 取出特征集
    y = data.iloc[:, data.columns == "Class"]        # 取出标签集
    return data, X, y
    
def subSampled(data):
    '''
    函数说明：
        对数据进行下采样处理
    参数：
        data : 数据集
    返回：
        under_sample_data ：下采样后得到的整个数据集
        X_undersample     ：下采样后得到的特征集
        y_undersample     ：下采样后得到的标签集
    '''
    
    # 获取标签中为1的长度/个数
    number_records_fraud = len(data[data.Class == 1]) # data的类型为DataFrame类型
    # 获取标签中为1的下标，并把列表转成array类型
    fraud_indices = np.array(data[data.Class == 1].index)
    # 获取标签中为0的下标(正常交易)，并把列表转成array类型
    normal_indices = data[data.Class == 0].index
    
    # 采用下采样(欠采样)方式处理数据,从正常的数据下标中选取不正常个数的下标
    # 即从正常数据(多)中随机抽取n个下标，使得正常的数据与不正常的数据的样本数相等
    random_normal_indices = np.random.choice(normal_indices, number_records_fraud,
                                             replace = False)
    random_normal_indices =np.array(random_normal_indices) # 转成Array数组
    
    # 把两个数据集的下标合并成一个
    under_sample_indices = np.concatenate([fraud_indices, random_normal_indices])
    
    # 抽取出下采样后的数据集,并分成特征集和标签集
    under_sample_data = data.iloc[under_sample_indices, :]
    X_undersample = under_sample_data.iloc[:, under_sample_data.columns != "Class"]
    y_undersample = under_sample_data.iloc[:, under_sample_data.columns == "Class"]
    
    # 打印正常以及别欺诈的数据的比重以及总数据的样本数
    # print("Percentage of normal transactions: ", len(under_sample_data[under_sample_data.Class == 0])/len(under_sample_data))
    # print("Percentage of fraud transactions: ", len(under_sample_data[under_sample_data.Class == 1])/len(under_sample_data))
    # print("Total number of transactions in resampled data: ", len(under_sample_data))
   
    return under_sample_data, X_undersample, y_undersample

def dataSegmentation(X, y, test_size = 0.3):
    '''
    函数说明 ：
        对数据进行切分，分成train和test两部分
    参数：
        X : 需要切分特征集
        y : 需要切分标签集
        test_size : test数据的大小，默认0.3
    返回：
        X_train : 切分后得到的用于训练的特征集
        X_test :  切分后得到的用于测试的特征集
        y_train : 切分后得到的用于训练的标签集
        y_test :  切分后得到的用于测试的标签集
    '''
    # 对整个数据集进行train : test = 7 : 3
    # 调用sklearn.cross_validation的train_test_split模块
    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                        test_size = test_size, 
                                                        random_state = 0)
    # 打印切分后的数据的长度，以及总长
    # print("Number transactions train dataset: ", len(X_train))
    # print("Number transactions test dataset: ", len(X_test))
    # print("Total number of transactions: ", len(X_train)+len(X_test))
    return X_train, X_test, y_train, y_test


# Recall = TP / (TP + FN)  召回率 = 真正 / (真正 + 错假 = 原数据的正的个数)
def printing_Kfold_scores(x_train_data, y_train_data, 
                          k=5, c_para_list = [0.01, 0.1, 1, 10, 100]):
    '''
    函数说明：
        进行k折交叉验证,并找出提供c参数中最优的值
    参数：
        x_train_data : 用于训练的特征集
        y_train_data : 用于训练的标签集
        k            : 折数,默认5折
        c_para_list  : 超参数C的可选值列表， 默认为[0.01, 0.1, 1, 10, 100]
    返回：
        best_c : 最优的超参数c
    '''
    # 对数据进行分层
    fold = KFold(len(y_train_data), k, shuffle=False)
    
    # 不同的c超参数值，C用在正则化项
    c_para_range = c_para_list
    results_table = pd.DataFrame(index = range(len(c_para_range), 2), 
                                columns = ['C_parameter', 'Mean recall score'])
    results_table["C_parameter"] = c_para_range # 把超参数c的列表加入到C_parameter列中
    # the k-fold will give 2 list: train_indices = indices[0], test_indices = indices[1]
    j = 0
    for c_param in c_para_range:
        print("------------------------------------")
        print("C parameter: ", c_param)
        print("------------------------------------")
        print("")
        
        recall_accs = []
        # 进行k折交叉验证
        for iteration, indices in enumerate(fold, start=1):
            # 实例化一个logistics regression 模型，采用l1正则化
            lr = LogisticRegression(C=c_param, penalty='l1')
            
            # Use the training data to fit the model, In this case, we use the 
            # portion of the fold to train the model with indices[0]. We the 
            # predict on the assigned as the 'test cross validation' with indices[1]
            lr.fit(x_train_data.iloc[indices[0], :], 
                   y_train_data.iloc[indices[0], :].values.ravel()) # .ravel()表示扁平化，变成向量
            
            # Predict values using the test indices in the training data
            # 用训练集中的验证部分进行预测
            y_pred_undersample = lr.predict(x_train_data.iloc[indices[1], :].values)
            
            # Calcuate the recall score and append it to a list for recall scores
            # representing the current c_parameter
            recall_acc = recall_score(y_train_data.iloc[indices[1], :].values,
                                       y_pred_undersample)
            recall_accs.append(recall_acc)
            print("Iteration ", iteration, ": recall score = ", recall_acc)
        
        # the mean value of those recall scores is the metric we want to 
        # save and get hold of.
        results_table.loc[j, 'Mean recall score'] = np.mean(recall_accs)
        j += 1              # 对当前的超参数c做完k折交叉验证，j增1
        print("")
        #打印当前c值的k折交叉验证的recall(召回率)值的平均值
        print("Mean recall score ", np.mean(recall_accs))
        print("")     
   
    '''
    The main problem:
    1) the type of "mean recall score" is object, you can't use "idxmax()" to calculate the value 
    2) you should change "mean recall score" from "object " to "float" 
    3) you can use apply(pd.to_numeric, errors = 'coerce', axis = 0) to do such things.
    '''
    # 获取最优的c值
    new = results_table.columns[results_table.dtypes.eq(object)] #get the object column of the best_c
    results_table[new] = results_table[new].apply(pd.to_numeric, errors = 'coerce', axis=0) # change the type of object best_c
    best_c = results_table.loc[results_table["Mean recall score"].idxmax()]["C_parameter"]
    
    #Finally, We can check which C parameter is the best amongst the chosen.
    print('********************************************************************************')
    print("Best model to choose from scross validation is with C parameter = ", best_c)
    print('********************************************************************************')
    
    return best_c

def plot_confusion_matrix(cnf_matrix, classes, title="Confusion matrix", cmap=plt.cm.Blues):
    '''
    函数说明：This function prints and plosts the confusion matrix.
             打印和画混淆矩阵
    参数：
        cnf_matrix ：需要绘画的混淆矩阵
        classes    : 类别(一般list类型)
        title      : 图形名称，默认为"Confusion matrix"
        cmap       ：颜色映射默认为plt.cm.Blues
    返回：
        无
    '''
    plt.imshow(cnf_matrix, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    '''
    plt.xticks([-1,0,1],['-1','0','1']) #第一个：对应X轴上的值，第二个：显示的文字
    '''
    plt.xticks(tick_marks, classes, rotation=0) # rotation:旋转
    plt.yticks(tick_marks, classes)
    
    thresh = cnf_matrix.max() / 2
    for i,j in itertools.product(range(cnf_matrix.shape[0]), range(cnf_matrix.shape[1])):
        plt.text(j, i, cnf_matrix[i, j], 
                 horizontalalignment='center',
                 color='white' if cnf_matrix[i, j] > thresh else 'black')
    '''
    补充知识点：https://matplotlib.org/api/pyplot_api.html
    matplotlib.pyplot.text(x, y, s, fontdict=None, withdash=False, **kwargs)
        x, y：表示坐标；
        s：字符串文本；
        fontdict：字典，可选；
        kwargs： 
            fontsize=12,
            horizontalalignment=‘center’、ha=’cener’
            verticalalignment=’center’、va=’center’
    fig.text()（fig = plt.figure(…)）
    ax.text() （ax = plt.subplot(…)）
    '''
    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")

if __name__ == "__main__":
    data = loadDataSet("creditcard.csv")  # 加载数据
    plotDataSet(data)                     # 图形显示数据
    # 由数据图可知，我们发现数据出现极度的类型不平衡
    # 一般对于这种情况，可以进行下采样和过采样处理，使得数据重新达到平衡
    data, X, y = dataPreProcess(data)     # 对数据进行预处理
    # 下采样方法：
    under_sample_data, X_underSample, y_underSample = subSampled(data)
    
    # 对整个数据集进行切分：train : test = 7:3
    X_train, X_test, y_train, y_test = dataSegmentation(X, y, test_size = 0.3)
    
    # 对整个数据集进行切分：train : test = 7:3
    X_under_train, X_under_test, \
    y_under_train, y_under_test = dataSegmentation(X_underSample, y_underSample, test_size = 0.3)
    
    ###########################################################################
    
    # 进行交叉验证选择最优的C参数,在下采样数据下
    best_c = printing_Kfold_scores(X_under_train, y_under_train, 5, [0.01, 0.1, 1, 10, 100])
   
    # 构建基于最优c参数的lLogistics Regression Model
    lr = LogisticRegression(C=best_c, penalty='l1')
    lr.fit(X_under_train, y_under_train.values.ravel()) # 训练模型
    y_pred_under = lr.predict(X_under_test.values) # 用训练好模型对测试进行预测
    
    # Compute confusion matrix，计算混淆矩阵
    cnf_matrix = confusion_matrix(y_under_test, y_pred_under)
    # 作用：确定浮点数字、数组、和numpy对象的显示形式。 
    np.set_printoptions(precision=2) # 精度为小数点后4位
    print("Recall metric in the testing dataset: ", 
      cnf_matrix[1, 1] / (cnf_matrix[1,0] + cnf_matrix[1, 1]))
    
    # plot non-normalized confusion matix
    class_name = [0, 1]
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=class_name, title="Confusion matrix -- under data")
    plt.show()   
    
    # 对整个完整数据的测试部分进行预测
    y_pred_all = lr.predict(X_test.values) # 用训练好模型对测试进行预测
    # Compute confusion matrix，计算混淆矩阵
    cnf_matrix = confusion_matrix(y_test, y_pred_all)  
    # 打印召回率
    print("Recall metric in the testing dataset: ", 
      cnf_matrix[1, 1] / (cnf_matrix[1,0] + cnf_matrix[1, 1]))  
    
    # plot non-normalized confusion matix
    class_name = [0, 1]
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=class_name, title="Confusion matrix -- all data")
    plt.show()    
    ###########################################################################
    
    ############################ 不做下采样处理训练预测 ########################
    best_c = printing_Kfold_scores(X_train, y_train)

    lr = LogisticRegression(C = best_c, penalty = 'l1')
    lr.fit(X_train,y_train.values.ravel())
    y_pred_undersample = lr.predict(X_test.values)

    # Compute confusion matrix
    cnf_matrix = confusion_matrix(y_test,y_pred_undersample)
    np.set_printoptions(precision=2)

    print("Recall metric in the testing dataset: ", cnf_matrix[1,1]/(cnf_matrix[1,0]+cnf_matrix[1,1]))
    
    # Plot non-normalized confusion matrix
    class_names = [0,1]
    plt.figure()
    plot_confusion_matrix(cnf_matrix
                          , classes=class_names
                          , title='Confusion matrix')
    plt.show()
    ###########################################################################
    
    ########################### 修改阀值进行对比 ###############################
    lr = LogisticRegression(C=0.01, penalty='l1')
    lr.fit(X_under_train, y_under_train.values.ravel()) # 训练模型
    '''
    假定在一个k分类问题中，测试集中共有n个样本。则：
    predict返回的是一个大小为n的一维数组，一维数组中的第i个值为模型预测第i个预测样本的标签；
    predict_proba返回的是一个n行k列的数组，第i行第j列上的数值是模型预测第i个预测样本的标签为j的概率。
                    此时每一行的和应该等于1。
    '''
    y_pred_undersample_proba = lr.predict_proba(X_under_test.values)
    print("y_pred_undersample_proba :", y_pred_undersample_proba)
    
    threshholds = [0.1, .2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    plt.figure(figsize = (10, 10))
    j=1
    for i in threshholds:
        y_test_predictions_high_recall = y_pred_undersample_proba[:, 1] > i
        plt.subplot(3,3,j)
        j += 1
        
        # Compute confusion matrix
        cnf_matrix = confusion_matrix(y_under_test, y_test_predictions_high_recall)
        np.set_printoptions(precision=2)
        
        print("Recall metric in the testing dataset: ",
              cnf_matrix[1, 1] / (cnf_matrix[1, 0] + cnf_matrix[1,1]))
        
        # Plot non-normalized confusion matrix
        class_names = [0, 1]
        plot_confusion_matrix(cnf_matrix, 
                              classes = class_name,
                              title="Threshold >= %s" % i)
    ###########################################################################
        
    ############################## 过采样 方法 ################################
    credit_cards=pd.read_csv('creditcard.csv')
    columns=credit_cards.columns
    # The labels are in the last column ('Class'). Simply remove it to obtain features columns
    features_columns=columns.delete(len(columns)-1)
    features=credit_cards[features_columns]
    labels=credit_cards['Class']
    features_train, features_test, \
    labels_train, labels_test = train_test_split(features, 
                                                 labels, 
                                                 test_size=0.2, 
                                                 random_state=0)
    oversampler=SMOTE(random_state=0)
    os_features,os_labels=oversampler.fit_sample(features_train,labels_train)
    os_features = pd.DataFrame(os_features)
    os_labels = pd.DataFrame(os_labels)
    best_c = printing_Kfold_scores(os_features,os_labels)    
    
    lr = LogisticRegression(C = best_c, penalty = 'l1')
    lr.fit(os_features,os_labels.values.ravel())
    y_pred = lr.predict(features_test.values)
    
    # Compute confusion matrix
    cnf_matrix = confusion_matrix(labels_test,y_pred)
    np.set_printoptions(precision=2)
    
    print("Recall metric in the testing dataset: ", cnf_matrix[1,1]/(cnf_matrix[1,0]+cnf_matrix[1,1]))
    
    # Plot non-normalized confusion matrix
    class_names = [0,1]
    plt.figure()
    plot_confusion_matrix(cnf_matrix
                          , classes=class_names
                          , title='Confusion matrix')
    plt.show()
    ###########################################################################
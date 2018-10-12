# -*- coding: utf-8 -*-
"""
Created on Wed Oct 10 20:00:35 2018

@author: 87955
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv("creditcard.csv")    # 读入数据
print(data.head())                      # 打印前5个样本

# 对class列的值的个数进行统计，即此处为统计1和0的各自出现的次数
count_classes = pd.value_counts(data["Class"], sort=True).sort_index()
count_classes.plot(kind = "bar") # 使用pd自带的画图函数数据进行可视化，采用条形图
plt.title("Fraud class histogram")
plt.xlabel("Class")
plt.ylabel("frequency")
print(count_classes)

from sklearn.preprocessing import StandardScaler

# 对data["Amount"]的数据进行标准化，并添加到data中
data["normAmount"] = StandardScaler().fit_transform(data["Amount"].values.reshape(-1, 1))
data = data.drop(["Time", "Amount"], axis=1) #drop掉属性Time和Amount的值，axis=1表示对列进行操作
print(data.head())
print("Data Type : ", type(data))
X = data.iloc[:, data.columns != "Class"] # 取出特征集
y = data.iloc[:, data.columns == "Class"] # 取出数据集

# 获取标签中为1的长度/个数
number_records_fraud = len(data[data.Class == 1]) 
# 获取标签中为1的下标，并把列表转成array类型
fraud_indices = np.array(data[data.Class == 1].index)
# 获取标签中为1的下标，并把列表转成array类型
normal_indices = data[data.Class == 0].index

# 采用下采样(欠采样)方式处理数据,从正常的数据下标中选取不正常个数的下标
# 即从正常数据(多)中随机抽取n个下标，使得正常的数据与不正常的数据的样本数相等
random_normal_indices = np.random.choice(normal_indices, number_records_fraud,
                                         replace = False)
random_normal_indices =np.array(random_normal_indices)

# 把两个数据集的下标合并成一个
under_sample_indices = np.concatenate([fraud_indices, random_normal_indices])

# 抽取出下采样后的数据集,并分成特征集和标签集
under_sample_data = data.iloc[under_sample_indices, :]
X_undersample = under_sample_data.iloc[:, under_sample_data.columns != "Class"]
y_undersample = under_sample_data.iloc[:, under_sample_data.columns == "Class"]



print("Percentage of normal transactions: ", len(under_sample_data[under_sample_data.Class == 0])/len(under_sample_data))
print("Percentage of fraud transactions: ", len(under_sample_data[under_sample_data.Class == 1])/len(under_sample_data))
print("Total number of transactions in resampled data: ", len(under_sample_data))


# 交叉验证
from sklearn.cross_validation import train_test_split

# 对整个数据集进行train : test = 7 : 3
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)
print("Number transactions train dataset: ", len(X_train))
print("Number transactions test dataset: ", len(X_test))
print("Total number of transactions: ", len(X_train)+len(X_test))

# 对下采样得到的数据集进行 7:3分割
X_train_undersample, X_test_undersample,\
y_train_undersample,y_test_undersample = train_test_split(X_undersample,
                                                          y_undersample,
                                                          test_size = 0.3,
                                                          random_state = 0)
print("")
print("Number transactions train dataset: ", len(X_train_undersample))
print("Number transactions test dataset: ", len(X_test_undersample))
print("Total number of transactions: ", len(X_train_undersample)+len(X_test_undersample))

# Recall = TP / (TP + FN)  召回率 = 真正 / (真正 + 错假 = 原数据的正的个数)
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import KFold, cross_val_score
from sklearn.metrics import confusion_matrix, recall_score, classification_report

def printing_Kfold_scores(x_train_data, y_train_data):
    fold = KFold(len(y_train_data), 5, shuffle=False)
    
    # 不同的c超参数值，C用在正则化项
    c_para_range = [0.01, 0.1, 1, 10, 100]
    
    results_table = pd.DataFrame(index = range(len(c_para_range), 2), 
                                columns = ['C_parameter', 'Mean recall score'])
    results_table["C_parameter"] = c_para_range
    # the k-fold will give 2 list: train_indices = indices[0], test_indices = indices[1]
    j = 0
    for c_param in c_para_range:
        print("------------------------------------")
        print("C parameter: ", c_param)
        print("------------------------------------")
        print("")
        
        recall_accs = []
        for iteration, indices in enumerate(fold, start=1):
            # 实例化一个logistics regression 模型
            lr = LogisticRegression(C=c_param, penalty='l1')
            
            # Use the training data to fit the model, In this case, we use the 
            # portion of the fold to train the model with indices[0]. We the 
            # predict on the assigned as the 'test cross validation' with indices[1]
            lr.fit(x_train_data.iloc[indices[0], :], 
                   y_train_data.iloc[indices[0], :].values.ravel())
            
            # Predict values using the test indices in the training data
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
        j += 1
        print("")
        print("Mean recall score ", np.mean(recall_accs))
        print("")     
   
    '''
    1) the type of "mean recall score" is object, you can't use "idxmax()" to calculate the value 
    2) you should change "mean recall score" from "object " to "float" 
    3) you can use apply(pd.to_numeric, errors = 'coerce', axis = 0) to do such things.
    '''
    new = results_table.columns[results_table.dtypes.eq(object)] #get the object column of the best_c
    results_table[new] = results_table[new].apply(pd.to_numeric, errors = 'coerce', axis=0) # change the type of object best_c
    best_c = results_table.loc[results_table["Mean recall score"].idxmax()]["C_parameter"]
    
    #Finally, We can check which C parameter is the best amongst the chosen.
    print('********************************************************************************')
    print("Best model to choose from scross validation is with C parameter = ", best_c)
    print('********************************************************************************')
    
    return best_c

best_c = printing_Kfold_scores(X_train_undersample, y_train_undersample)

import itertools

def plot_confusion_matrix(cm, classes, title="Confusion matrix", cmap=plt.cm.Blues):
    '''
    函数说明：This function prints ans plosts the confusion matrix.
    '''
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=0)
    plt.yticks(tick_marks, classes)
    
    thresh = cm.max() / 2
    for i,j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j], 
                 horizontalalignment='center',
                 color='white' if cm[i, j] > thresh else 'black')
    
    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    

lr = LogisticRegression(C=best_c, penalty='l1')
lr.fit(X_train_undersample, y_train_undersample.values.ravel())
y_pred_undersample = lr.predict(X_test_undersample.values)

# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test_undersample, y_pred_undersample)
np.set_printoptions(precision=2)

print("Recall metric in the testing dataset: ", 
      cnf_matrix[1, 1] / (cnf_matrix[1,0] + cnf_matrix[1, 1]))

# plot non-normalized confusion matix
class_name = [0, 1]
plt.figure()
plot_confusion_matrix(cnf_matrix,
                      classes=class_name,
                      title="Confusion matrix")

plt.show()            


lr = LogisticRegression(C=best_c, penalty='l1')
lr.fit(X_train_undersample, y_train_undersample.values.ravel())
y_pred = lr.predict(X_test.values)

# Compute confusion on matrix
cnf_matrix = confusion_matrix(y_test, y_pred)
np.set_printoptions(precision=2)

print("Recall metric in the testing dataset: ", 
      cnf_matrix[1,1] / (cnf_matrix[1,1] + cnf_matrix[1, 0]))

# Plot non-normalized confusion matrix
class_name = [0, 1]
plt.figure()
plot_confusion_matrix(cnf_matrix, 
                      class_name, 
                      title="Confusion matrix")
plt.show()

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



lr = LogisticRegression(C=0.01, penalty='l1')
lr.fit(X_train_undersample, y_train_undersample.values.ravel())
y_pred_undersample_proba = lr.predict_proba(X_test_undersample.values)
print("y_pred_undersample_proba :", y_pred_undersample_proba)

threshholds = [0.1, .2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
plt.figure(figsize = (10, 10))
j=1
for i in threshholds:
    y_test_predictions_high_recall = y_pred_undersample_proba[:, 1] > i
    plt.subplot(3,3,j)
    j += 1
    
    # Compute confusion matrix
    cnf_matrix = confusion_matrix(y_test_undersample, y_test_predictions_high_recall)
    np.set_printoptions(precision=2)
    
    print("Recall metric in the testing dataset: ",
          cnf_matrix[1, 1] / (cnf_matrix[1, 0] + cnf_matrix[1,1]))
    
    # Plot non-normalized confusion matrix
    class_names = [0, 1]
    plot_confusion_matrix(cnf_matrix, 
                          classes = class_name,
                          title="Threshold >= %s" % i)



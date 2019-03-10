import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

data = pd.read_csv('train.csv')

train_size = 10000
test_size = 1000

trainX = np.array(data.iloc[0:train_size,1:])
trainX = trainX / 255.0
trainy = data.iloc[0:train_size,0]

testX = np.array(data.iloc[train_size:train_size + test_size,1:])
testX = testX / 255.0
testy = data.iloc[train_size:train_size + test_size,0]
'''
show_list = np.random.randint(low=0,high=train_size,size=10)
for i in show_list:
    sample = trainX[i])
    sample = np.reshape(sample,(28,28))
    plt.imshow(sample,cmap='gray')
    plt.show()
'''          
svm_model = LinearSVC(max_iter=5000)
logit_model = LogisticRegression(max_iter=1000)
decision_tree = DecisionTreeClassifier()

svm_model.fit(trainX,trainy)
logit_model.fit(trainX,trainy)
decision_tree.fit(trainX,trainy) 

print('The svm scored: %f' % svm_model.score(testX,testy))
print('The logit_model scored: %f' % logit_model.score(testX,testy))
print('The decision tree scored: %f' % decision_tree.score(testX,testy))

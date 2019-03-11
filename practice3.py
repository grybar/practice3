import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report

def create_train_and_test_sets(data,train_size=1,test_size=1):
    data = data.sample(frac=1).reset_index(drop=True)
    trainX = np.array(data.iloc[0:train_size,1:])
    trainX = trainX / 255.0
    trainy = np.array(data.iloc[0:train_size,0])

    testX = np.array(data.iloc[train_size:train_size + test_size,1:])
    testX = testX / 255.0
    testy = np.array(data.iloc[train_size:train_size + test_size,0])
    return trainX,trainy,testX,testy

data = pd.read_csv('train.csv')

train_size = 10000
test_size = 1000

trainX,trainy,testX,testy = create_train_and_test_sets(data,train_size,test_size)
print(testX.shape)
print(testy.shape)
'''
show_list = np.random.randint(low=0,high=train_size,size=10)
for i in show_list:
    sample = trainX[i]
    sample = np.reshape(sample,(28,28))
    plt.imshow(sample,cmap='gray')
    plt.show()
'''          
svm_model = LinearSVC(max_iter=10000)
#logit_model = LogisticRegression(max_iter=1000)
#decision_tree = DecisionTreeClassifier()

svm_model.fit(trainX,trainy)
#logit_model.fit(trainX,trainy)
#decision_tree.fit(trainX,trainy) 

print('The svm scored: %f' % svm_model.score(testX,testy))
#print('The logit_model scored: %f' % logit_model.score(testX,testy))
#print('The decision tree scored: %f' % decision_tree.score(testX,testy))
'''
train_size = 1000
test_size = 1000

trainX_small,trainy_small,testX_small,testy_small = create_train_and_test_sets(data,train_size,test_size)

svm_model_less_train_img = LinearSVC(max_iter=5000)
svm_model_less_train_img.fit(trainX_small,trainy_small)

print('The svm with 1000 train samples scored: %f' % svm_model_less_train_img.score(testX_small,testy_small))
'''
predy = svm_model.predict(testX)
classification_report(testy,predy)

show_list = np.random.randint(low=0,high=test_size,size=20)
fig=plt.figure(figsize=(10,10))
for i in range(1,21):
    sample = testX[show_list[i-1]]
    sample = np.reshape(sample,(28,28))
    fig.add_subplot(5,4,i)
    plt.imshow(sample)
    plt.title(label='Prediction = {0} Actual = {1}'.format(predy[show_list[i-1]],testy[show_list[i-1]]),
              fontsize='small')
    
plt.show()

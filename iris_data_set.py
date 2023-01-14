from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler,OneHotEncoder
import numpy as np
from sklearn.model_selection import train_test_split
from keras.activations import softmax
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import  precision_score


#import dataset into Sklearn
li=load_iris()

x=li.data
y=li.target

scale=MinMaxScaler()
x1=scale.fit_transform(x)

ohe=OneHotEncoder()
y1=ohe.fit_transform(y[:,np.newaxis]).toarray()
y2=y1.astype('int')

#split test and train dataset
xtrain,xtest,ytrain,ytest=train_test_split(x1,y2,test_size=0.2)


#create model 
model=Sequential()
model.add(Dense(12,activation='relu',input_dim=4))
model.add(Dense(12,activation='relu'))
model.add(Dense(3,activation=softmax))
model.compile(optimizer='adam',loss='categorical_crossentropy')

#fit or train model 
model.fit(xtrain,ytrain,epochs=100)

#now Predict model
ypred=model.predict(xtest)
ypred>0.45

for i in range(len(ypred)):
  j=np.argmax(ypred[i,:])
  ypred[i,0]=0
  ypred[i,1]=0
  ypred[i,2]=0
  ypred[i,j]=1

ypred.astype("int")

#Apply cross validetion in model using Confusion metrics

 # this is precision
p_sum=0
for i in range(len(ypred)):
  a=precision_score(ytest[i],ypred[i])
  p_sum=p_sum+a
p_sum=p_sum/len(ypred)
p_sum

#this is recall score
from sklearn.metrics import  recall_score
r_sum=0
for i in range(len(ypred)):
  a=recall_score(ytest[i],ypred[i])
  r_sum=r_sum+a
r_sum=r_sum/len(ypred)
r_sum

# this is f1 score
f1=(2*p_sum*r_sum)/p_sum*r_sum
f1



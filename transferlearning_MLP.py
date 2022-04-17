import pandas as pd
import numpy as np
data1 = pd.read_csv('transfer_data1.csv')

X = data1.loc[:,'x']
y = data1.loc[:,'y']

from matplotlib import pyplot as plt
fig1 = plt.figure()
plt.scatter(X,y)
plt.title('y vs x data1')
plt.xlabel('x')
plt.ylabel('y')
#plt.show()

X = np.array(X).reshape(-1,1)

from keras.models import Sequential
from keras.layers import Dense
model1 = Sequential()
model1.add(Dense(units=50,input_dim=1,activation='relu'))  #input_dim输入的维度是1
model1.add(Dense(units=50,activation='relu'))
#输出层
model1.add(Dense(units=1,activation='linear'))
model1.summary()

model1.compile(optimizer='adam',loss='mean_squared_error')

model1.fit(X,y,epochs=500)

y_predict = model1.predict(X)
from sklearn.metrics import r2_score
r2 = r2_score(y,y_predict)
print("epochs=500:",r2)

fig2 = plt.figure()
plt.scatter(X,y,label='raw data')
plt.scatter(X,y_predict,label='predict result')
plt.title('y vs x data1, epochs=500')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
#plt.show()

#模型训练 500+500=1000
model1.fit(X,y,epochs=500)

y_predict = model1.predict(X)
from sklearn.metrics import r2_score
r2 = r2_score(y,y_predict)
print(r2)
fig3 = plt.figure()
plt.scatter(X,y,label='raw data')
plt.scatter(X,y_predict,label='predict result')
plt.title('y vs x data1, epochs=1000')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
#plt.show()

#模型训练 500+500+1000=2000
model1.fit(X,y,epochs=1000)

y_predict = model1.predict(X)
from sklearn.metrics import r2_score
r2 = r2_score(y,y_predict)
print(r2)
fig4 = plt.figure()
plt.scatter(X,y,label='raw data')
plt.scatter(X,y_predict,label='predict result')
plt.title('y vs x data1, epochs=2000')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
#plt.show()

#模型存储***
import joblib
joblib.dump(model1,'transferlearning.m')

#模型加载
model2=joblib.load('transferlearning.m')
model2.summary()

data2 = pd.read_csv('transfer_data2.csv')

X_new = data2.loc[:,'x']
y_new = data2.loc[:,'y']

X_new=np.array(X_new).reshape(-1,1)   #X_new唯独确认
y_new_predict = model2.predict(X_new)
r2_new = r2_score(y_new,y_new_predict)
print(r2_new)

fig5 = plt.figure()
plt.scatter(X,y,label='raw data1')
plt.scatter(X_new,y_new,label='raw data2')
plt.plot(X_new,y_new_predict,label='predict result data2',c='r')
plt.title('y_new vs X_new data1')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
#plt.show()

#模型迁移学习
model2.fit(X_new,y_new,epochs=20)

y_new_predict = model2.predict(X_new)
r2_new = r2_score(y_new,y_new_predict)
print(r2_new)
fig6 = plt.figure()
plt.scatter(X,y,label='raw data1')
plt.scatter(X_new,y_new,label='raw data2')
plt.plot(X_new,y_new_predict,label='predict result data2',c='r')
plt.title('y_new vs X_new data1, transferlearning_epochs=20')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
#plt.show()

#模型迁移学习
model2.fit(X_new,y_new,epochs=80)

y_new_predict = model2.predict(X_new)
r2_new = r2_score(y_new,y_new_predict)
print(r2_new)
fig7 = plt.figure()
plt.scatter(X,y,label='raw data1')
plt.scatter(X_new,y_new,label='raw data2')
plt.plot(X_new,y_new_predict,label='predict result data2',c='r')
plt.title('y_new vs X_new data1, transferlearning_epochs=80')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
#plt.show()

plt.show(fig1, fig2, fig3, fig4)
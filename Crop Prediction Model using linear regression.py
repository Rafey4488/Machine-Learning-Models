#Datasets can be downloaded from here
#https://drive.google.com/drive/folders/1-qu9HupJ7K9wVrxwCTJIUa2Y1d9ynlyp?usp=sharing

import csv
import numpy as np
import seaborn as sns
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.model_selection import train_test_split

dataset2  = pd.read_csv('/content/drive/MyDrive/ML Dataset/cropph.csv')
dataset  = pd.read_csv('/content/drive/MyDrive/ML Dataset/production.csv')
dataset1  = pd.read_csv('/content/drive/MyDrive/ML Dataset/regressiondb.csv')

cropdict = {'Bajra' :1 ,'Banana':2,'Barley':3,'Bean':4,'Black pepper':5,'Blackgram':6,'Bottle Gourd':7,'Brinjal':8,
'Cabbage':9,'Cardamom':10,'Carrot':11,'Castor seed':12,'Cauliflower':13,'Chillies':14,
'Colocosia':15,'Coriander':16,'Cotton':17,'Cowpea':18,'Drum Stick':19,'Garlic':20,
'Ginger':21,'Gram':22,'Grapes':23,'Groundnut':24,'Guar seed':25,'Horse-gram':26,
'Jowar':27,'Jute':27,'Khesari':28,'Lady Finger':29,'Lentil':30,'Linseed':31,'Maize':32,
'Mesta':33,'Moong(Green Gram)':34,'Moth':35,'Onion':36,'Orange':37,'Papaya':38,
'Peas & beans (Pulses)':39,'Pineapple':40,'Potato':41,'Raddish':42,'Ragi':43,
'Rice':44,'Safflower':45,'Sannhamp':46,'Sesamum':47,'Soyabean':48,'Sugarcane':49,
'Sunflower':50,'Sweet potato':51,'Tapioca':52,'Tomato':53,'Turmeric':54,'Urad':55,
'Varagu':56,'Wheat':57}


dataset1['Cropconversion'] = dataset1['Cropconversion'].map(cropdict)

a=dataset1.Cropconversion
'''
print("****************************************************")
print(dataset1.head())
print(dataset1.shape)
print(dataset1.index)
print(list(dataset1.columns))
print("****************************************************")
print(dataset.head())
print(dataset.shape)
print(dataset.index)
print(list(dataset.columns))
lm1 = smf.ols(formula='Production ~ Rainfall+Temperature+Ph', data=dataset1).fit()
print(lm1.params)'''
#linear_reression
X1 = dataset[['Rainfall', 'Temperature', 'Ph']]
Y1 = dataset.Production
# Split data
X1_train, X1_test, y1_train, y1_test = train_test_split(X1, Y1, random_state=1)
# Instantiate model
lm_2 = LinearRegression(fit_intercept=True)
# Fit Model
lm_2.fit(X1_train, y1_train)
# Predict
y1_pred = lm_2.predict(X1_test)
# RMSE
rmse = np.sqrt(metrics.mean_squared_error(y1_test, y1_pred))
print('Root Mean Square error is: ',rmse)
linear_model_1 = smf.ols(formula='Production ~ Rainfall+Temperature+Ph', data=dataset).fit()
print(linear_model_1.params)
sns.pairplot(dataset, x_vars=['Rainfall','Temperature','Ph'], y_vars='Production', height=7, aspect=0.7,kind='reg')


x = dataset['Temperature']
y = dataset.Production
n = np.size(x)
x_mean = np.mean(x)
y_mean = np.mean(y)
x_mean,y_mean
Sxy = np.sum(x*y)- n*x_mean*y_mean
Sxx = np.sum(x*x)-n*x_mean*x_mean
b1 = Sxy/Sxx
b0 = y_mean-b1*x_mean
print('slope b1 is', b1)
print('intercept b0 is', b0)
plt.scatter(x,y)
plt.xlabel('Independent variable X')
plt.ylabel('Dependent variable y')

y_pred = 0.4 * x + -5.4
plt.scatter(x, y, color = 'blue')
plt.plot(x, y_pred, color = 'red')
plt.xlabel('X')
plt.ylabel('y')

error = y - y_pred
se = np.sum(error**2)
print('squared error is', se)
mse = se/n
print('mean squared error is', mse)
rmse = np.sqrt(mse)
print('root mean square error is', rmse)
SSt = np.sum((y - y_mean)**2)
R2 = 1- (se/SSt)
print('R square is', R2)


x = dataset['Ph']
y = dataset.Production
n = np.size(x)
x_mean = np.mean(x)
y_mean = np.mean(y)
x_mean,y_mean
Sxy = np.sum(x*y)- n*x_mean*y_mean
Sxx = np.sum(x*x)-n*x_mean*x_mean
b1 = Sxy/Sxx
b0 = y_mean-b1*x_mean
print('slope b1 is', b1)
print('intercept b0 is', b0)
plt.scatter(x,y)
plt.xlabel('Independent variable X')
plt.ylabel('Dependent variable y')

y_pred = 5.9 * x + -33
plt.scatter(x, y, color = 'blue')
plt.plot(x, y_pred, color = 'red')
plt.xlabel('X')
plt.ylabel('y')

error = y - y_pred
se = np.sum(error**2)
print('squared error is', se)
mse = se/n
print('mean squared error is', mse)
rmse = np.sqrt(mse)
print('root mean square error is', rmse)
SSt = np.sum((y - y_mean)**2)
R2 = 1- (se/SSt)
print('R square is', R2)




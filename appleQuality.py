
#Kaggle = https://www.kaggle.com/code/umutsefkansak/apple-quality-prediction-with-logistic-regression/notebook

#%% Libraries
import pandas as pd
import numpy as np


#%% Data

data = pd.read_csv("apple_quality.csv")

data.info()
print("total null values: ",data.isnull().sum().sum())


#%% Dropping null and useless values

# Drop null values
data.dropna(inplace=True)

#Drop useless values
data.drop(["A_id"],axis = 1 , inplace = True)

# converting object values to numeric
data['Acidity'] = pd.to_numeric(data['Acidity'], errors='coerce')
data.info()
#%%
# converting object values to numeric
data.Quality = [0 if quality == "bad" else 1 for quality in data.Quality]
#%% 

#x and y
y = data.Quality.values
x = data.drop(["Quality"],axis = 1)

#%%
# Train test split
from sklearn.model_selection import train_test_split

x_train,x_test, y_train,y_test = train_test_split(x, y,test_size=0.2,random_state=42)

x_train = x_train.T
x_test = x_test.T
y_train = y_train.T
y_test = y_test.T


#%% 
# Initalizing weight and bias
def initialize(dimension):
    
    w = np.full((dimension,1),0.01)
    b = 0.0
    return w,b

#%%
# Sigmoid function
def sigmoid(z):
    
    y_head = 1/(1+np.exp(-z))
    return y_head


#%%
# Forward and backward propagation
def forward_backward_prediction(w,b,x_train,y_train):
    
    #Forward propagation
    z = np.dot(w.T,x_train) + b
    y_head = sigmoid(z)
    loss = -y_train*np.log(y_head)-(1-y_train)*np.log(1-y_head)
    cost = (np.sum(loss))/x_train.shape[1]
    
    #Backward propagation
    derivative_weight = (np.dot(x_train,((y_head-y_train).T)))/x_train.shape[1]
    derivative_bias = np.sum(y_head-y_train)/x_train.shape[1]
    
    gradients = {"derivative_weight":derivative_weight,"derivative_bias":derivative_bias}
    
    return cost,gradients

#%%
# Updating weight and bias
def update(w,b,x_train,y_train,learning_rate,number_of_iterations):
    
    for i in range(number_of_iterations):
        
        cost,gradients = forward_backward_prediction(w, b, x_train, y_train)
        
        w = w - learning_rate*gradients["derivative_weight"]
        b = b - learning_rate*gradients["derivative_bias"]
        
        if i % 10 == 0:
            print(i," iteration cost: ",cost)
    
    parameters = {"weight":w,"bias":b}
    
    return parameters

#%%
# Prediction function
def prediction(w,b,x_test):
    
    y_heads = sigmoid(np.dot(w.T,x_test)+b)
    y_predictions = np.zeros((1,x_test.shape[1]))
    
    for i in range(y_heads.shape[1]):
        
        if y_heads[0,i] <= 0.5:
            y_predictions[0,i] = 0
        else:
            y_predictions[0,i] = 1
    
    return y_predictions

#%%
# Logistic regression model
def logistic_regression(x_train,x_test,y_train,y_test,learning_rate,number_of_iterations):
    
    dimension = x_train.shape[0]
    w,b = initialize(dimension)
    
    parameters = update(w, b, x_train, y_train, learning_rate, number_of_iterations)
    
    y_predictions = prediction(parameters["weight"], parameters["bias"], x_test)
    
    print("test accuracy: {} %".format(100 - np.mean(np.abs(y_predictions - y_test)) * 100))


#%%
# Testing
logistic_regression(x_train, x_test, y_train, y_test, 1,45)
    

    
#%%
# Logistic regression with sklearn library
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()
lr.fit(x_train.T,y_train.T)

print("Score: ",lr.score(x_test.T,y_test.T))

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

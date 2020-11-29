# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 13:28:06 2019

@author: User
"""
import Menu
import numpy as np
import pandas as pd
import datetime
import pandas_datareader.data as web
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing, svm
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn import neighbors
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler


#mpl.rcParams['backend'] = "qt4agg"
#mpl.rcParams['backend.qt4'] = "PySide"

class PredictiveAnalyticsMenu(Menu.Menu):
    #Must have the self variable as a parameter  as this is a class method not a free function
    #Self is an instance of the class
    #To call this method we must first instantiate an object of the class
    def displayOptions(self):
        choice = input("What would you like to do 1. Linear Regression \n Q to quit\n")
        if(choice=="1"):
            self.linearRegression()
        elif(choice.upper()=="Q"):
            print("You have exited the system")
        else:
            print("Incorrect input, must be in range specified")
    

    
    def preProcessData(self, forecast_out):
        df['Prediction'] = df['Adj Close'].shift(-forecast_out) 
        X = np.array(df.drop(['Prediction'], 1))
        X = X[:-forecast_out] 
        y = np.array(df['Prediction'])
        y = y[:-forecast_out]  
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.15)
        
        return X_train, X_test, y_train, y_test
    
    #Use Linear Regression model to predict future stock prices 
    
    def linearRegression(self):
        df = self.getStockData()
        close_px = df['Adj Close']
        #determines how far in the future we will predict values for i.e. n = 30 would be 30 days
        forecast_out = int(input("How many days in the future would you like predicted?"))
        df['Prediction'] = df['Adj Close'].shift(-forecast_out)
        #Converts the df to a numpy array
        X = np.array(df.drop(['Prediction'], 1))
        #Remove last n rows 
        X = X[:-forecast_out]
        y = np.array(df['Prediction'])
        #Remove NaN values i.e. last n rows 
        #returns everything except the last n items
        y = y[:-forecast_out]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.15)
        
        lr = LinearRegression()
        print(X_train)
        print(y_train)
        #train model
        lr.fit(X_train, y_train)
    
        confidenceScore = lr.score(X_test, y_test)
        print("R squared score for this model is: ", confidenceScore)
        
        y_predict = lr.predict(X_test)
        print("RMSE is ", np.sqrt(metrics.mean_squared_error(y_test, y_predict)))

        
        #Gets last forecast_out numbers in array
        x_forecast = np.array(df.drop(['Prediction'], 1))[-forecast_out:]
        lr_prediction = lr.predict(x_forecast)
        for counter, value in enumerate(lr_prediction):
            print("Day: ", counter+1)
            print("Expected stock price: ", value)
        
        plt.plot(lr_prediction)
        #lr_prediction.plot(label="Predicted stock prices")
        #plt.legend()
        

        
        
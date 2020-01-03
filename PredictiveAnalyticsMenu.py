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
        choice = input("What would you like to do 1. Linear Regression \n Q to quit")
        if(choice=="1"):
            self.linearRegression()
        elif(choice=="2"):
            self.SVM()
        elif(choice.upper()=="Q"):
            print("You have exited the system")
        else:
            print("Incorrect input, must be in range specified")
    

    
    def preProcessData(self, forecast_out):
        df['Prediction'] = df['Adj Close'].shift(-forecast_out)
        
        #Create independent data set X
        #This dataset will be used for training the model
        #Converts the df to a numpy array
        X = np.array(df.drop(['Prediction'], 1))
        #Remove last n rows 
        X = X[:-forecast_out]
        #print(X)
        
        #Create dependent data set Y
        #This is the target data which will hold price predictions
        #It is a typical convention within ML programming to denote X as the features as y as coreesponding labels
        y = np.array(df['Prediction'])
        #Remove NaN values i.e. last n rows 
        #returns everything except the last n items
        y = y[:-forecast_out]
        
        #Split the data into 85% train and 15% test
        #Model will train by taking input train_x and learning to match to train_y
        #Then the model will attempt to predict an accurate test_y based on train y
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.15)
        
        return X_train, X_test, y_train, y_test
    
    def linearRegression(self):
        df = self.getStockData()
        close_px = df['Adj Close']
        #determines how far in the future we will predict values for i.e. n = 30 would be 30 days
        forecast_out = int(input("How many days in the future would you like predicted?"))
        #Create 'Prediction' datafram colume which stores the predicted price for the stock n days in the future
        #Date remains as index
        df['Prediction'] = df['Adj Close'].shift(-forecast_out)
        print(df.head())
        #Create independent data set X
        #This dataset will be used for training the model
        #Converts the df to a numpy array
        X = np.array(df.drop(['Prediction'], 1))
        #Remove last n rows 
        X = X[:-forecast_out]
        #Create dependent data set Y
        #This is the target data which will hold price predictions
        #It is a typical convention within ML programming to denote X as the features as y as coreesponding labels
        y = np.array(df['Prediction'])
        #Remove NaN values i.e. last n rows 
        #returns everything except the last n items
        y = y[:-forecast_out]
        
        #Split the data into 85% train and 15% test
        #Model will train by taking input train_x and learning to match to train_y
        #Then the model will attempt to predict an accurate test_y based on train y
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.15)
        
        lr = LinearRegression()
        print(X_train)
        print(y_train)
        #train model
        lr.fit(X_train, y_train)
        
        #crossValScore = cross_validate(estimator=svr, X=X_train, y=y_test,cv=10)
        #Calculated co-efficient of determinanation i.e. r squared
        #Score function compares predictions for X_test with y_test and return r squared value
        confidenceScore = lr.score(X_test, y_test)
        print("R squared score for this model is: ", confidenceScore)
        
        y_predict = lr.predict(X_test)
        #Calculate model Root Mean Squared Error (RMSE)
        #RMSE answers the question: how similar are the numbers in list 1 to list 2 on average
        #In this case we want to get the similarity between the values predicted by the model on test data y_test and the actual data
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
        
    #K Nearest Neighbour algorithm
    #Based on the independent variables KNN finds similarities between old data points and new
    #KNN can be used for both regression and clasification problems
    #The K in KNN is the number of nearest neighbours we take a vote from
    #As we decrease the value of K to 1 our prediction becomes less stable
    #1. Load Data
    #2. Initlialise value of k
    #3. Iterate 1 to total number of training data points (training data represents a known point)
    # a.Calculate distance between test data and each row of training data
    def KNN(self):
        scaler = MinMaxScaler(feature_range(0,1) )
        x_train, X_test, y_train, y_test = preProcessData()
        x_train_scaled = scaler.fit_transformed(x_train)
        
        
 #   def SVM(self):
# =============================================================================
#         df = self.getStockData()
#         close_px = df['Adj Close']
#         #determines how far in the future we will predict values for i.e. n = 30 would be 30 days
#         forecast_out = int(input("How many days in the future would you like predicted?"))
#         #Create 'Prediction' datafram colume which stores the predicted price for the stock n days in the future
#         #Date remains as index
#         df['Prediction'] = df['Adj Close'].shift(-forecast_out)
#         print(df.head())
#         #Create independent data set X
#         #This dataset will be used for training the model
#         #Converts the df to a numpy array
#         X = np.array(df['Adj Close'])
#         #Remove last n rows 
#         X = X[:-forecast_out]
#         #Create dependent data set Y
#         #This is the target data which will hold price predictions
#         #It is a typical convention within ML programming to denote X as the features as y as coreesponding labels
#         y = np.array(df['Prediction'])
#         #Remove NaN values i.e. last n rows 
#         #returns everything except the last n items
#         y = y[:-forecast_out]
#         
#         #Split the data into 85% train and 15% test
#         #Model will train by taking input train_x and learning to match to train_y
#         #Then the model will attempt to predict an accurate test_y based on train y
#         X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.15)
#         #Create support vector machine
#         svr = svm.SVR()
#         
#         #train model
#         svr.fit(X_train, y_train)
#         
#         #crossValScore = cross_validate(estimator=svr, X=X_train, y=y_test,cv=10)
#         confidenceScore = svr.score(X_test, y_test)
#         print("Confidence score for SVM is: ", confidenceScore)
# =============================================================================
        
        
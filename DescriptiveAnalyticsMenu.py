import pandas as pd
import datetime
import pandas_datareader.data as web
from pandas import Series, DataFrame
import matplotlib.pyplot as plt
from matplotlib import style
from scipy.stats import linregress
import matplotlib as mpl
import tkinter 
import Menu

#By using import Menu we have the ability to call functions using Menu.function() which makes it clear what we are calling and what library we are using
#from matplotlib import * would be worst format as we would use function1() to call a function however if another function1()
#is declared it willl override the import

from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk)
# Implement the default Matplotlib key bindings.
from matplotlib.backend_bases import key_press_handler
from matplotlib.figure import Figure

import numpy as np
import company_stock_info

#Adjusting matplotlib
mpl.rc('figure', figsize=(8,7))
style.use('ggplot')
#mpl.rcParams['backend'] = "qt4agg"
#mpl.rcParams['backend.qt4'] = "PySide"
class DescriptiveAnalyticsMenu(Menu.Menu):
    def displayOptions(self):
        choice = input("Descriptive Analytics Menu \n What would you like to see? \n 1. Moving Average \n 2. Returns \n 3. Raw Time Series \n 4. Weighted Moving Average \n 5. MACD \n")
        if(choice=="1"):
            #movingAverage function is within the class so to call it the call must be prefaced with the class name
            #Self is the class instance
            self.movingAverage()
        
        elif(choice=="2"):
            self.expectedReturn()
            
        elif(choice=="3"):
            self.rawTimeSeries()
        elif(choice=="4"):
            self.weightedMovingAverage()
        elif(choice=="5"):
            self.MACD()
        elif(choice.upper()=="Q"):
            print("You have quit the system")
        else:
            print("Incorrect input. Must be one of options specified")
    
    
    
    def rawTimeSeries(self):
        df =self.getStockData()
        
        close_px = df['Adj Close']
        close_px.plot(label = "AAPL")
        plt.legend()
    
    #Rolling Mean/Moving Average smooths out price data by creating a constantly updated average price
    #The moving average acts as resistance meaning from the downtrend and uptrend of stocks you could expect it will follow the trend and will be less likely to deviate from resistance point
    def movingAverage(self):#x is the list
        df = self.getStockData()
        
        #df = company_stock_info.filter_dataset(x, stock_code)
        
        #print(df)
        close_px = df['Adj Close']
     
        #window is the size of the moving window
        #Loops through the dataframe and gets the windows of size 100, calculates mean for each window
        #Moving average steadily rises over the window2012
        mavg = close_px.rolling(window =100).mean()
        print(mavg)
        
        
        
        
        
        #draw_graph(close_px, mavg)

        

        #Moving average can be used to predict when to buy/sell stocks 
        #i.e. sell during downturn buy during upturn
        close_px.plot(label = "AAPL")
        mavg.plot(label="Moving Average")
        plt.legend()
        #draw_graph(close_px, mavg)

    
    def weightedMovingAverage(self):
        df = self.getStockData()
        close_px = df['Adj Close']
        numOfDays = int(input("For what window (i.e. number of days) would you like a weighted moving average (WMA) calculated?"))
        #Adjust = False specifies that we want to use recursive calculation mode
        #EMA (Exponential Moving Average) reduces lag in traditional MA by putting more weight on recent observations
        #Span is provided by the user and the function automatically generates the decay
        wmavg = close_px.ewm(span=numOfDays, adjust=False).mean()
        
        close_px.plot(label = "AAPL")
        wmavg.plot(label="Exponential Moving Average")
        plt.legend()
        #plt.waitforbuttonpress()
    
    #Calculate Moving Average Convergence Divergence
    #MACD is a trend following momentumm indicator that shows the relationship between 2 moving averages of a security's price
    #MACD Formula: 12 day EMA - 26 day EMA
    def MACD(self):
        df = self.getStockData()
        close_px = df['Adj Close']  
        df['12 ema'] = close_px.ewm(span=12, adjust=False).mean()
        df['26 ema'] = close_px.ewm(span=26, adjust=False).mean()
        
        df['MACD'] = (df['12 ema'] - df['26 ema'])
        close_px.plot(label = "AAPL")
        df['MACD'].plot(label = "MACD")
        plt.legend()
        #plt.waitforbuttonpress()
    
    
    #Expected return measures the mean, or expected value, of the probability distribution of investment returns 
    #Ideal stocks should return as high and stable as possible
    def expectedReturn(self):
        df = self.getStockData()
        
        close_px = df['Adj Close']
        
        #df.shift(i) shifts the entire dataframe i units down
        #This gets return i/return i-1
        rets = close_px/close_px.shift(1) - 1
        rets.plot(label='return')
        #plt.waitforbuttonpress()
    
    #if __name__ =="__main__":
    #    displayOptions()
    def draw_graph(x, y):
        root = tkinter.Tk()
        root.wm_title("Graph")
    
        fig = Figure(figsize=(6, 6), dpi=100)
        fig.add_subplot(111).plot(x, label="Hello")
        fig.add_subplot(111).plot(y, label="Howya")
        fig.legend()
        
        canvas = FigureCanvasTkAgg(fig, master=root)  # A tk.DrawingArea.
        canvas.draw()
        canvas.get_tk_widget().pack(side=tkinter.TOP, fill=tkinter.BOTH, expand=1)
        button = tkinter.Button(master=root, text="Quit", command= root.destroy)
        button.pack(side=tkinter.BOTTOM)
        
        toolbar = NavigationToolbar2Tk(canvas, root)
        toolbar.update()
        canvas.get_tk_widget().pack(side=tkinter.TOP, fill=tkinter.BOTH, expand=1)

        tkinter.mainloop()

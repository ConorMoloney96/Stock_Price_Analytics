# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 22:53:15 2019

@author: User
"""
import datetime
import pandas_datareader.data as web

#Superclass/Parent class which all menu classes will inherit from
class Menu:
    def validate_date(self, date_text):
        try:
            datetime.datetime.strptime(date_text, '%Y-%m-%d')
            return date_text
        except ValueError:
            return "Error"
    
    def getStockData(self, dates = None, stock_code = None):
        if(dates is None and stock_code is None):
            stock = input("What Stock would you like to analyze? ")
            start_date = "Error"
            while(start_date is "Error"):
                start_date = input('What would you like as start date? \n Date should be in YYYY-MM-DD format:  ')
                start_date = self.validate_date(start_date)
            start_year, start_month, start_day = map(int, start_date.split('-'))
            end_date = "Error"
            while(end_date is "Error"):
                end_date = input('What would you like as end date? \n Date should be in YYYY-MM-DD format:  ')
                end_date = self.validate_date(end_date)
            end_year, end_month, end_day = map(int, end_date.split('-'))
            #start_year = int(input("What year would you like as start year?"))
            #end_year = int(input("What year would you like as end year?"))
            start_time = datetime.datetime(start_year, start_month, start_day)
            end_time = datetime.datetime(end_year, end_month, end_day)
        
        else:
            print("Start date", dates[0])
            print("End date", dates[1])
            start_time = dates[0]
            end_time = dates[1]
        #Download Apple stocks from Yahoo API for the given times
        df = web.DataReader(stock, 'yahoo', start_time, end_time)
        #print(df)
        return df

    #Abstract method that must be implemented by inherited classes
    def displayOptions(self):
        raise NotImplementedError
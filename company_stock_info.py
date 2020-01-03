# -*- coding: utf-8 -*-

# Used for obtaining stock information

import datetime
import os.path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



def get_company_names():
    # Reading in the csv file and adjusting to suit our needs
    comp_names = pd.read_csv("https://old.nasdaq.com/screening/companies-by-name.aspx?letter=0&exchange=nasdaq&render=download")
    comp_names = pd.read_csv("companylist.csv")

    comp_names = comp_names.loc[:, ~comp_names.columns.str.contains('^Unnamed')]
    comp_names.set_index('Symbol',inplace=True)
    comp_names["Name"] = comp_names["Name"].str.lower()


    return comp_names


def get_company_stock_info(users_input):
    
    # Our API key is 4RX66NA329CAVYU6
    print("THE USER INPUT")
    print(users_input)
    
    comp_stock_info = pd.read_csv('https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol={}&outputsize=full&apikey=4RX66NA329CAVYU6&datatype=csv'.format(users_input.lower()))
    # comp_stock_info.set_index('timestamp', inplace=True)
    
    
    return comp_stock_info


def get_basic_statistics(x, stock_code):# x is a list of dates
    
    
    df = filter_dataset(x, stock_code)
    
    basic_stats = dict()
    
    basic_stats['Mean'] = df['adjusted_close'].mean()
    basic_stats['Median'] = df['adjusted_close'].median()
    basic_stats['1stQ'] = df['adjusted_close'].quantile(0.25)
    basic_stats['3rdQ'] = df['adjusted_close'].quantile(0.75)
    basic_stats['Inter_Quartile_Range'] = (df['adjusted_close'].quantile(0.75) - df['adjusted_close'].quantile(0.25))
    basic_stats['Max'] = df['adjusted_close'].max()
    basic_stats['Min'] = df['adjusted_close'].min()
    basic_stats['Range'] = (df['adjusted_close'].max() - df['adjusted_close'].min())
    basic_stats['Standard_Deviation'] = df['adjusted_close'].std()
    basic_stats['Coef_of_variation'] = (df['adjusted_close'].std()) / df['adjusted_close'].mean()
    basic_stats['Variance'] = df['adjusted_close'].var()
    
    
    return basic_stats
    
    
   
    
def filter_dataset(x, stock_code):# x is a list of dates
    print("The stock code: {}".format(stock_code))
    start_date = x[0].strftime("%Y-%m-%d")
    end_date = x[1].strftime("%Y-%m-%d")
   
    
    data = get_company_stock_info(stock_code)
    data['timestamp'] = data['timestamp'].astype(str)
    
    # data.set_index('timestamp')
    
    # data.set_index('timestamp', inplace=True)
    print(data.info())
    # data['timestamp'] = pd.to_datetime(data['timestamp']) 
    # masked = (sample['timestamp'] > start_date) & (sample['timestamp'] <= end_date)
    
    masked = (data['timestamp'] >= start_date) & (data['timestamp'] <= end_date)
    
    filtered_data = data.loc[masked]
    
    #if !(filtered_data.empty):
        # do something
    
    return filtered_data

# df1 = get_company_stock_info("zg")
# print(df1)
# print("Whats above this")




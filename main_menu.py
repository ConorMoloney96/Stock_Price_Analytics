# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 12:17:17 2019

@author: User
"""
import DescriptiveAnalyticsMenu
import PredictiveAnalyticsMenu
import Menu

class main_menu(Menu.Menu):
    #Driver method
    #Allows users to go into other menus
    def displayOptions():
        choice = ""
        choice = input("This is the Main Menu. \n What would you like to do? \n 1. Descriptive Analytics \n 2. Predictive Analytics \n Press Q to Quit \n")
        if(choice=="1"):
            DAMenu = DescriptiveAnalyticsMenu.DescriptiveAnalyticsMenu()
            DAMenu.displayOptions()
        elif(choice=="2"):
            PAMenu = PredictiveAnalyticsMenu.PredictiveAnalyticsMenu()
            PAMenu.displayOptions()
        elif(choice.upper()=="Q"):
           print("You have decided to quit the program")
        else:
            print("Incorrect input. Must be one of the options specified")
        
if __name__ =="__main__":
    main_menu.displayOptions()
import sys
import os 
from colorama import Fore, Back, Style

def selectZScoreMethod():
    print(Fore.BLUE, end='')
    print("\nEnter '0' to use Modified Z-Score(Default) or '1' to use normal Z-Score Method for Signatures")
    print(Style.RESET_ALL, end='')
    zScoreMethod = input("=> ")
    return zScoreMethod

def calculate_outliers():
    print(Fore.BLUE, end='')
    print("\nEnter the complete file name with path")
    print(Style.RESET_ALL, end='')
    file_name = input("=> ")
    zScoreMethod = selectZScoreMethod()
    os.system('python3 cal_outlier.py -j '+file_name+' '+zScoreMethod)

def calculate_outliers_acls():
    print(Fore.BLUE, end='')
    print("\nEnter the complete file name with path")
    print(Style.RESET_ALL, end='')
    file_name = input("=> ")
    zScoreMethod = selectZScoreMethod()
    os.system('python3 cal_outlier.py -j '+file_name+' '+zScoreMethod+' -a')

def edit_outliers():
    print(Fore.BLUE, end='')
    print("\nEnter the complete Named-Structure file name with path")
    print(Style.RESET_ALL, end='')
    file_name = input("=> ")
    print(Fore.BLUE, end='')
    print("\nEnter the complete WhiteList File name with path")
    print(Style.RESET_ALL, end='')
    whitelistFileName = input("=> ")
    print(Fore.BLUE, end='')
    print()
    os.system('python3 cal_outlier.py -e '+file_name+' '+whitelistFileName)

def view_outliers():
    print(Fore.BLUE, end='')
    print("\nEnter the complete file name with path")
    print(Style.RESET_ALL, end='')
    file_name = input("=> ")
    os.system('python3 cal_outlier.py -d '+file_name)

def deviant_plot():
    print(Fore.BLUE, end='')
    print("\nEnter the complete file name with path")
    print(Style.RESET_ALL, end='')
    file_name = input("=> ")
    os.system('python3 view_analysis.py -p '+file_name)

def score_plot():
    print(Fore.BLUE, end='')
    print("\nEnter the complete file name with path")
    print(Style.RESET_ALL, end='')
    file_name = input("=> ")
    os.system('python3 view_analysis.py -s '+file_name)

def choose_action():
    action_selected = 0
    options_count = 7
    while action_selected <= options_count:
        
        print("==========================")
        print("Choose Action:\n")
        print("1 => CALCULATE OUTLIERS - ACL_Lists")
        print("2 => CALCULATE OUTLIERS - ACLs")
        print("3 => EDIT OUTLIERS")
        print("4 => VIEW OUTLIER NODES")
        print("5 => DEVIANT PROPERTIES PLOT")
        print("6 => SCORING COMPARISION PLOT")
        print("7 => EXIT APPLICATION")
        print("==========================")

        action_selected = input("\nAction => ")
        action_selected = int(action_selected)

        if action_selected == 1:
            print(Fore.RED, end='')
            print("*Calculate Outliers Selected*")
            print(Style.RESET_ALL, end='')
            calculate_outliers()
        elif action_selected == 2:
            print(Fore.RED, end='')
            print("*Calculate Outliers Selected*")
            print(Style.RESET_ALL, end='')
            calculate_outliers_acls()
        elif action_selected == 3:
            print(Fore.RED, end='')
            print("*Edit Outliers Selected*")
            print(Style.RESET_ALL, end='')
            edit_outliers()
        elif action_selected == 4:
            print(Fore.RED, end='')
            print("*View Outliers Selected*")
            print(Style.RESET_ALL, end='')
            view_outliers()
        elif action_selected == 5:
            print(Fore.RED, end='')
            print("*Deviant Properties Plot Selected*")
            print(Style.RESET_ALL, end='')
            deviant_plot()
        elif action_selected == 6:
            print(Fore.RED, end='')
            print("*Scoring Comparision Plot Selected*")
            print(Style.RESET_ALL, end='')
            score_plot()
        elif action_selected == 7:
            print(Fore.BLUE, end='')
            print("Exiting Function ...\n")
            print(Style.RESET_ALL, end='')
            sys.exit()
        else:
            print(Fore.BLUE, end='')
            print("Invalid Action Selected\nExiting Function ...\n")
            print(Style.RESET_ALL, end='')
            sys.exit()

# Calling the function
choose_action()
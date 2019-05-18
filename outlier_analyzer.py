import sys
import os 
from colorama import Fore, Back, Style

def calculate_outliers():
    print(Fore.BLUE, end='')
    print("\nEnter the complete file name with path")
    print(Style.RESET_ALL, end='')
    file_name = input("=> ")
    print()
    os.system('python3 cal_outlier.py -j '+file_name)

def calculate_outliers_acls():
    print(Fore.BLUE, end='')
    print("\nEnter the complete file name with path")
    print(Style.RESET_ALL, end='')
    file_name = input("=> ")
    print()
    os.system('python3 cal_outlier.py -j '+file_name+' -a')

def edit_outliers():
    print(Fore.BLUE, end='')
    print("\nEnter the complete file name with path")
    print(Style.RESET_ALL, end='')
    file_name = input("=> ")
    print(Fore.BLUE, end='')
    print("\nEnter the KEY")
    print(Style.RESET_ALL, end='')
    key = input("=> ")
    print(Fore.BLUE, end='')
    print("\nEnter the VALUE")
    print(Style.RESET_ALL, end='')
    value = input("=> ")
    print()
    os.system('python3 cal_outlier.py -e '+file_name+' '+key+' '+value)


def view_outliers():
    print(Fore.BLUE, end='')
    print("\nEnter the complete file name with path")
    print(Style.RESET_ALL, end='')
    file_name = input("=> ")
    os.system('python3 cal_outlier.py -d '+file_name)


def choose_action():
    action_selected = 0
    while action_selected < 4:
        
        print("==========================")
        print("Choose Action:\n")
        print("1 => CALCULATE OUTLIERS - ACL_Lists")
        print("2 => CALCULATE OUTLIERS - ACLs")
        print("3 => EDIT OUTLIERS")
        print("4 => VIEW OUTLIER NODES")
        print("5 => EXIT APPLICATION")
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
import sys
import os 
from pathlib import Path
from colorama import Fore, Back, Style
#import cal_outlier

import os.path
from os import path

# Print INFO
def printINFO(info):
    print(Fore.BLUE, end='')
    print("#",info)
    print(Style.RESET_ALL, end='')

# Error Message
def printError(errormsg, arg):
    print("****************************")
    print(Fore.RED, end='')
    print(errormsg,":", arg)
    print(Style.RESET_ALL, end='')
    print("****************************")

def selectZScoreMethod():
    printINFO("Enter '0' to use Modified Z-Score(Default) or '1' to use normal Z-Score Method for Signatures")
    zScoreMethod = input("=> ")
    return zScoreMethod

def calculate_outliers():
    printINFO("Enter the complete file name with path")
    file_name = input("=> ")
    if (not path.exists(file_name)):
        printError("Input valid file name (with absolute path)",file_name)
        return False
    zScoreMethod = selectZScoreMethod()
    os.system('python3 cal_outlier.py -j '+file_name+' '+zScoreMethod)

def calculate_outliers_acls():
    printINFO("Enter the complete file name with path")
    file_name = input("=> ")
    if (not path.exists(file_name)):
        printError("Input valid file name (with absolute path)",file_name)
        return False
    zScoreMethod = selectZScoreMethod()
    os.system('python3 cal_outlier.py -j '+file_name+' '+zScoreMethod+' -a')

def edit_outliers():
    printINFO("Enter the complete Named-Structure file name with path")
    file_name = input("=> ")
    printINFO("Enter the complete WhiteList File name with path")
    whitelistFileName = input("=> ")
    print(Fore.BLUE, end='')
    print()
    os.system('python3 cal_outlier.py -e '+file_name+' '+whitelistFileName)

def view_outliers():
    printINFO("Enter the complete file name with path")
    file_name = input("=> ")
    os.system('python3 cal_outlier.py -d '+file_name)

def deviant_plot():
    printINFO("Enter the complete file name with path")
    file_name = input("=> ")
    os.system('python3 view_analysis.py -p '+file_name)

def score_plot():
    printINFO("Enter the complete file name with path")
    file_name = input("=> ")
    os.system('python3 view_analysis.py -s '+file_name)

def run_ranking_module():
    printINFO("\nEnter the Network Directory path for which you want to run Severity Module")
    network_directory = input("=> ").strip()
    node_rank_file = network_directory+"/node_ranks.json"
    curr_directory = os.path.dirname(os.path.abspath(__file__))
    outlier_directory_path = curr_directory+"/outliers_"+network_directory.split("/")[-1]
    ranking_file_path = str(Path(curr_directory).parent)+'/ranking-severity/ranking.py'
    os.system('python3 '+ranking_file_path+' '+node_rank_file+' '+outlier_directory_path)

def main():
    action_selected = 0
    #options_count = 8
    while action_selected != 8:
        
        print("==========================")
        print("Choose Action:\n")
        print("1 => CALCULATE OUTLIERS (e.g., Composed ACL_List level)")
        print("2 => CALCULATE OUTLIERS (e.g., Discrete ACLs)")
        print("3 => EDIT OUTLIERS")
        print("4 => VIEW OUTLIER NODES")
        print("5 => DEVIANT PROPERTIES PLOT")
        print("6 => SCORING COMPARISION PLOT")
        print("7 => OUTLIERS/BUG SEVERITY & RANKING")
        print("8 => EXIT APPLICATION")
        print("==========================")

        action_selected = input("\nAction => ")
        try:
            action_selected = int(action_selected)
        except:
            print("Invalid argument selected. Returning to main menu.")

        if action_selected == 1:
            printINFO("Option: Calculate Outliers Selected !!\n")
            if not calculate_outliers():
                continue

        elif action_selected == 2:
            printINFO("Option:  Calculate Outliers Selected !!\n")
            calculate_outliers_acls()

        elif action_selected == 3:
            printINFO("Option: Edit Outliers Selected !!\n")
            edit_outliers()

        elif action_selected == 4:
            printINFO("Option: View Outliers Selected !!\n")
            view_outliers()

        elif action_selected == 5:
            printINFO("Option: Deviant Properties Plot Selected !!\n")
            deviant_plot()

        elif action_selected == 6:
            printINFO("Option: Scoring Comparision Plot Selected !!\n")
            score_plot()

        elif action_selected == 7:
            printINFO("Option: Ranking Module Selected !!\n")
            run_ranking_module()

        elif action_selected == 8:
            printError("Exiting Function. Action Selected:", action_selected)
            sys.exit()

        else:
            printINFO("Invalid Action Selected. Returning to Main menu...\n")

# Calling main function
main()
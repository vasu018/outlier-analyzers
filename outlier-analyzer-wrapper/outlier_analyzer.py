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
    """
    Ask users for their choice of Z-score to use.
    Default 0 for Modified Z-Score using median (generally for skewed distribution of data);
    1 for Z-score using mean (for normal distribution of data);
    Input:
        0 for Modified Z-score.
        1 for Z-score.
    Output:

    """
    printINFO("Enter '0' to use Modified Z-Score(Default) or '1' to use normal Z-Score Method for Signatures")
    zScoreMethod = input("=> ")
    return zScoreMethod


def calculate_outliers():
    """
        Calculate Outliers in ACL_Lists.
    Input:
        User action 1 and the full ptah location to calculate outliers for.
    Output:
        Outliers calculated for ACL_Lists and a results file.
    """

    printINFO("Enter the complete file name with path")
    file_name = input("=> ")
    if (not path.exists(file_name)):
        printError("Input valid file name (with absolute path)",file_name)
        return False
    zScoreMethod = selectZScoreMethod()
    os.system('python3 cal_outlier.py -j '+file_name+' '+zScoreMethod)


def calculate_outliers_acls():
    """
    Calculate Outliers in ACLs.
    Input:
        User action 2 and the full path location to calculate outliers for.
    Output:
        Outliers calculated for ACLs and a results file.
    """

    printINFO("Enter the complete file name with path")
    file_name = input("=> ")
    if (not path.exists(file_name)):
        printError("Input valid file name (with absolute path)",file_name)
        return False
    zScoreMethod = selectZScoreMethod()
    os.system('python3 cal_outlier.py -j '+file_name+' '+zScoreMethod+' -a')


def edit_outliers():
    """
    Edit the outliers found in the ACL to re-tune the signature and calculate new outliers.
    Input:
        User action 3, named-structure file path and WhiteList file path.
    Output:
        New outliers found from re-tune the signature based on updates from the user.
    """

    printINFO("Enter the complete Named-Structure file name with path")
    file_name = input("=> ")
    printINFO("Enter the complete WhiteList File name with path")
    whitelistFileName = input("=> ")
    print(Fore.BLUE, end='')
    print()
    os.system('python3 cal_outlier.py -e '+file_name+' '+whitelistFileName)


def view_outliers():
    """
    Displays all the outlier-nodes found.
    Input:
        User action 4 with the results file path.
    Output:
        Outliers displayed.
    """
    printINFO("Enter the complete file name with path")
    file_name = input("=> ")
    os.system('python3 cal_outlier.py -d '+file_name)


def deviant_plot():
    """
    Plots the frequency of deviant properties.
    Input:
        User action 5 and the results file path.
    Output:
        A graph containing the frequency of deviant properties.
    """

    printINFO("Enter the complete file name with path")
    file_name = input("=> ")
    os.system('python3 view_analysis.py -p '+file_name)


def score_plot():
    """
    Graphs deviant_score, similarity_score, overall_signature_score.
    Input:
        User action 6 and results file path.
    Output:
        Graph containing various scores calculated when trying to catch outliers.
    """
    printINFO("Enter the complete file name with path")
    file_name = input("=> ")
    os.system('python3 view_analysis.py -s '+file_name)


def run_ranking_module():
    """
    Ranks the severity of the outliers based on three metrics.
    Metric1 - Threshold based.
    Metric2 - Uses the Pagerank Algorithm.
    Metric3 - By building a feature dependency graph.
    final_score of outlier = node_score of outlier * prob_score of outlier.
    Input:
        The directory path to perform the ranking analysis on.
    Output:
        A ranking hierarchy of all the bugs found.
    """
    printINFO("\nEnter the Network Directory path for which you want to run Severity Module")
    network_directory = input("=> ").strip()
    node_rank_file = network_directory+"/node_ranks.json"
    curr_directory = os.path.dirname(os.path.abspath(__file__))
    outlier_directory_path = curr_directory+"/"+network_directory.split("/")[-1]
    ranking_file_path = str(Path(curr_directory).parent)+'/ranking-severity/ranking.py'
    print(ranking_file_path, node_rank_file, outlier_directory_path)

    os.system('python3 '+ranking_file_path+' '+node_rank_file+' '+outlier_directory_path)


def main():
    action_selected = 0
    while action_selected != 8:
        actions = ['Calculate Outliers - Composite ACL_List Level', 'Calculate Outliers - Discrete ACL Level',
                   'Edit Outliers', 'View Outliers', 'Plot deviant properties', 'Plot scoring comparison',
                   'Rank outliers/bug severity', 'Exit Application']

        action_functions = ['calculate_outliers()', 'calculate_outliers_acls()', 'edit_outliers()', 'view_outliers()',
                            'deviant_plot()', 'score_plot()', 'run_ranking_module()']

        print("===========")
        print("Choose Action:\n")
        for index, action in enumerate(actions, 1):
            print("Choose %d to %s" % (index, action))
        print("===========")

        action_selected = input("\nAction => ")
        try:
            action_selected = int(action_selected)
        except:
            print("Invalid argument selected. Returning to main menu.")

        info_string = actions[action_selected - 1] + " selected!!"
        printINFO(info_string)

        eval(action_functions[action_selected - 1])


# Calling main function
if __name__ == '__main__':
    main()
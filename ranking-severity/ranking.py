###############################################################################
# This script is used to calculate the severity of the outiers and rank them accoridingly. It also accepts user input in 
# whether a particular outlier is a bug or FP and udpates the ranking accordingly.

#running the script:

#python ranking.py <node_ranks.json(file with PageRank scores of all the nodes)> <directory of all outliers json files>

#example:
#python ranking.py node_ranks.json XYZ_outliers

###############################################################################33

import pandas as pd
import numpy as np
from sklearn import preprocessing
from collections import defaultdict
import logging
import random
import scipy
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import seaborn as sns
import json
import os
from IPython.display import display
import sys
from colorama import Fore, Back, Style

# Reading the node_ranks.json file and storing the data in a dictionary.
with open(sys.argv[1], 'r') as fp:
    ranks = json.load(fp)

# Helper function to calculate metric_2 score of outliers
def calc_m2(nodes):
    s = 0
    for node in nodes:
        try:
            s = max(s,ranks[node])
        except KeyError:
            s = s
    return s  

# Reading all the outlier files in the directory 
directory = sys.argv[2]
try:
	files = os.listdir(directory)
except:
	print("Directory path incorrect")

print("JSON files gathered, loading the data")
json_files=[]

# Feature scores which we calculated seperately
feature_scores = {'Routing_Policy': 0.10256905808871317,
 'Route_Filter_List': 0.1590760837298577,
 'IP_Access_List': 0.11335485818142918,
 'IKE_Phase1_Proposals': 0.125,
 'AS_Path_Access_List': 0.125,
 'IKE_Phase1_Policies': 0.125,
 'IKE_Phase1_Keys': 0.125,
 'IPsec_Phase2_Policies': 0.125,
 'VRF': 0.125}

# We iterate all the outlier files and process them seperately.
# At the end, we will merge all the outliers in a single dataframe.
for file in files:
    print("Processing the file: ", file)
    # Loading the data in a dataframe.
    try:
        df1 = pd.read_json(os.path.join(directory,file), orient='split')
    except:
        continue
	
	# Calculating metric 1 for all outliers
    df1['Metric1'] = df1['similarity_score']/df1["acl_score"]
	# Removing the entries which have similarity_score = 1, as they are not bugs.
    df1 = df1[df1['Metric1']<1]  
    # If there are no outliers left, move on to the next file.    
    if len(df1)==0:
        continue
	# Calculating metric2 using the helper functions.
	# Metric 2 takes into account how well connected a node, which has the particular outlier, is to the other nodes and 
	# gives a score to the outlier on this basis.
    df1['Metric2'] = df1.apply(lambda row: calc_m2(row['nodes']), axis=1)
              
    file_name = file[8:-5]
    df1['Metric3'] = feature_scores[file_name]
    
	# Scaling both metric 1 and meric 2.
    x = df1[['Metric1']] #returns a numpy array
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    df1['Metric1'] = x_scaled
    
    x = df1[['Metric2']] #returns a numpy array
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    df1['Metric2'] = x_scaled
    
    x = df1[['Metric3']] #returns a numpy array
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    df1['Metric3'] = x_scaled
   
	# Calculating the severity score
    df1['Severity_score'] = df1['Metric1']+df1['Metric2']+df1['Metric2']

	# Renaming the cluster number of each outlier. The new cluster number will be unique across all named structures.
	# The format of the new cluster number will be = "named_structure" + "_" + "original cluster number"
	# Example = IP_Access_List_12, VRF_2
    df1['cluster_name'] = file.split(".")[0]+"_" + df1['cluster_number'].astype(str)

	#Appending the dataframe to a list of dataframes.
    json_files.append(df1)

print("Processig done")
# Merging the list of dataframes .  
df = pd.concat(json_files)

# Assigning an outlier_id to each entry. This will be used later in metric 4 when the user has to specify whether the outlier is a bug or FP.
df['Outlier_id'] = list(range(0,len(df)))

# Sorting the dataframe in descending order by their severity score.
df=df.sort_values(by = "Severity_score",ascending = False)

# Creating a dictionary which will store which outliers belong to the same clusters.
clusters = defaultdict(list)

print("Saving clusters")
def extract_clusters(row):
    clusters[row['cluster_name']].append(row['Outlier_id'])
df.apply(lambda row:extract_clusters(row),axis = 1)

# Helper function to update the scores when the user marks an outlier as a bug.
def found_bug(index,df=df):
    try:
		# We get the cluster_number of the outlier which has been marked as a bug.
        cluster_number = df.loc[df["Outlier_id"]==index,"cluster_name"].iloc[0]
    except:
        print("Index out of range")
        return

	# Get all the outliers which belong to that cluster.
    cluster = clusters[cluster_number]

	# Update the scores of all these outliers by increasing their severity scores by 10%.	
    for i in cluster:
        df.loc[df.Outlier_id==i,'Severity_score'] = df.loc[df.Outlier_id==i,'Severity_score']*1.1

# Helper function to update the scores when the user marks an outlier as a FP.
def found_fp(index,df=df):
    try:
		# We get the cluster_number of the outlier which has been marked as FP.
        cluster_number = df.loc[df["Outlier_id"]==index,"cluster_name"].iloc[0]
    except:
        print("Index out of range")
        return

	# Get all the outliers which belong to that cluster.
    cluster = clusters[cluster_number]

	# Update the scores of all these outliers by decreasing their severity scores by 10%.	
    for i in cluster:
        df.loc[df.Outlier_id==i,'Severity_score'] = df.loc[df.Outlier_id==i,'Severity_score']*0.9

# Delete all the columns which are no longer required.
del df['similarity_score']
del df['acl_score']
del df['max_sig_score']
del df['Metric1']
del df['Metric2']
del df['Metric3']
del df['cluster_number']

# Helper function to display the list of outliers.
def display_dataframe(df):
    columns = df.columns
    for i in range(len(df)):
        print(Fore.GREEN)
        print("############################################################")
        print("OUTLIER NO ",i+1)
        print(Style.RESET_ALL) 
        for col in columns:
            print(Fore.RED)
            print(col,":")
            print(Style.RESET_ALL) 
            print(df.iloc[i][col])

# We run a loop where the user can examine the outliers and mark them as a bug or FP.
while True:
    print("Choose action:")
    print("1 = Display top outliers")
    print("2 = Found Bug")
    print("3 = Found FP")
    print("4 = Display all outliers")
    print("5 = Exit")
    print("Action->",end = " ")
    x = int(input())
    if x == 1:
        print("Top 5 outliers:")
        display_dataframe(df.head())
    elif x ==2:
        print("Enter Outlier_id of the bug")
        index = int(input())
        found_bug(index)
        df = df.sort_values(by="Severity_score",ascending = False)
        print("Scores updated")
    elif x ==3:
        print("Enter Outlier_id of the FP")
        index = int(input())
        found_fp(index)
        df = df.sort_values(by="Severity_score",ascending = False)
        print("Scores updated")
    elif x==4:
        print("All outliers:")
        display_dataframe(df)
    elif x ==5:
        print("Exiting application")
        break
    else:
        print("Invalid action")


# This code generates the list of outliers which require manual inspection to determine whether they are bugs or FPs. 
# It is used in creating ground truth for the tool. The input of the tool is the json file of outliers. 
# The output is a json file of list of outliers which require manual inspection.
# The tool can be run as: python ground_truth_creation.py json_file_name. The input file must be in the same folder as the script.
# The resulting json file will be saved with the same name as the input file, with "_Inspection" appended at the end.
# For example, if the input file name is XYZ_outliers.json, the output file will be XYZ_outliers_Inspection.json.

import pandas as pd
import numpy as np
from collections import defaultdict
import logging
import random
import scipy
import matplotlib.pyplot as plt
import json
import numpy as np
import os
from scipy.cluster.vq import kmeans2 as kmeans
import math
import sys

# Import the outlier file which you want to work on and load it in a dataframe.
file = sys.argv[1]
try:
    df = pd.read_json(file,orient = "split")
except:
    print("Please enter a valid name for the file which you want to use as an argument when running the code. The file must be in the same directory as this script.")
    sys.exit(0)
df = df.reset_index()

# Calculate the ratio of similarity score and max_sig_score and sort the dataframe by that score in descending order.
df['Score'] = df['similarity_score']/df['max_sig_score']
df = df.sort_values(by = ['Score'],ascending = False)

#Exclude the ones which have score of 1, as they match with the signature and hence are not bugs.
df = df[df['Score']<1]

# The input file contains outliers for multiple signatures. For example, if we get 10 signatures in our signature based outlier  detection logic, there will be 10 groups of outliers in the input file.
# The input data contains a column called cluster_number, which signifies which signature an outlier is a part of. 
# If there are 10 signatures, the cluster_number column will have values from 0-9, signifying which signature the particular outlier belongs to.
# We will split the dataframe based on this column so that we can work with outliers belonging to a particular signature at one time.

final_df = []
for cluster in df.cluster_number.unique():# df.cluster_number.unique() will give us the list of unique entries in the column.
    temp = df[df["cluster_number"]==cluster]
    # temp will now hold outliers belonging to a particular signature.
    y = list(temp.Score)     # getting the similarity scores for the outliers.
    
    #We now cluster the outliers into 3 groups on basis of their similarity scores using kmeans with k=3. The 3 groups will be: bugs, FPs and require manual inspection.
    # To do this, we seed the kmeans with the initial centroids. The three centroids are the first, last and median values of the similarity scores (sorted in descending order).
    _,labels = kmeans(y,k=[y[0],y[len(y)//2],y[-1]],minit="matrix")
    # labels give us which cluster (out of the 3) the particular outlier belongs to. Label value 0 = BUG, 1= requires manual inspection, 2=FP.
    
    # Add the labels as a column in the dataframe.
    temp['Labels'] = labels

    # Append the dataframe to a list of dataframes.
    final_df.append(temp)

#final_df is a list of dataframes which we will now merge back together.
final_df = pd.concat(final_df)

# filter out the outliers which require manual inspection (label = 1)
manual_inspection = final_df[final_df['Labels']==1]

# store it as json file for further use.
fileName = file.split(".")[0]+"_Inspection.json"
manual_inspection.to_json(fileName)
print("Output file saved as: " ,fileName)
	

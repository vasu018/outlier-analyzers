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
import numpy as np
import os
from scipy.cluster.vq import kmeans2 as kmeans
import math

# Import the outlier file which you want to work on.
file = "outliers_IP_Access_List.json"
df = pd.read_json(file)
df = df.reset_index()

# Calculate the ratio fo similarity score and max_sig_score which will be between 0 and 1.
df['Score'] = df['similarity_score']/df['max_sig_score']
df = df.sort_values(by = ['Score'],ascending = False)

#Exclude the ones which have score of 1.
df = df[df['Score']<1]
y = list(df.Score)

# Plot for similarity score.
# plt.plot(y, len(y) * [1],".")
# plt.yticks([])
# plt.xlabel("Similarity score")
# plt.title(file)
# plt.savefig(file+".png")

# Perform K means on the similarity scores with pre-defined k(3).
# K means will give us a label for every outlier. The corresponding tags of the labels are:
# 0 = bug
# 1 = requires manual inspection
# 2 = FP
centroid,labels = kmeans(y,k=[y[0],y[len(y)//2],y[-1]],minit="matrix")

# Add the labels to the dataframe and store it as json file for further use.
df['Labels'] = labels
df.to_json(file+"labelled.json")

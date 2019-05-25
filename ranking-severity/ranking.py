###############################################################################
#running the script:

#python ranking.py <node_ranks.py(file with PageRank scores of all the nodes)> <list of all outliers json files>

#example:
#python ranking.py node_ranks.json signature-ip-access.json signature-route-filter.json signature-route-policy.json

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
import sys


## Here, we are accessing all the json files mentioned in the command line arguments, and loading the data into dataframes.
## Before final merging of all dataframes, we perform the required operations to calculate the different metrics and the severity score of all outliers.
## We caclulate the metric 2 based on the pageRank score of the nodes, and meric3 score using the staticticaal approach.
with open(sys.argv[1], 'r') as fp:
    ranks = json.load(fp)

num_files = len(sys.argv)-2
nodes_set = set()
def store_nodes(nodes):
    s = []
    ret = 0
    for node in nodes:
        ret = max(ret,ranks[node])
    return ret

def score_df_sum(node_score,score,df_score):
    print("df_score",node_score,score,df_score)
    df_score[0] += (node_score*score)

def score_df_mean(node_score,score,df_score):
    #print("df_score",node_score,score,df_score)
    df_score.append(node_score*score)

def calc_m2(nodes):
    s = []
    for node in nodes:
        s.append(ranks[node])
    return np.median(sorted(s[int(len(s)/2):]))+np.mean(s)
def get_name(outlier):
    return outlier['name']

json_files =[]
for file in sys.argv[2:]:
    df = pd.read_json(file,orient = "split")
    df['outlier_id'] = df.apply(lambda row: get_name(row['outlier_names']), axis=1)
    df = df[df['score_ratio']<=0.7]
    df_score =[]
    df['Metric_1'] = 1 - df['score_ratio']
    df['node_scores'] = df.apply(lambda row: store_nodes(row['outlier_nodes']), axis=1)
    df.apply(lambda row: score_df_mean(row['node_scores'],1-row['score_ratio'],df_score), axis=1)
    df["Metric_3"] = np.mean(df_score)*100
    df['Metric_2'] = df.apply(lambda row: calc_m2(row['outlier_nodes']), axis=1)
    x = df[['Metric_2']] #returns a numpy array
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    df['Metric_2'] = x_scaled
    del df['node_scores']
    json_files.append(df)


## This dataframe will contain all the outliers with all the relevant info.(metric scores, severity scores)
df = pd.concat(json_files)
df['severity_score'] = df['Metric_3']*(df['Metric_1'] + df['Metric_2'])


## The following code is for incorporating the user feedback into the severity calculationself.
# We start by building populations of correleated outliers, and whenever any outier is marked as a bug or false-positive, we update
# the other outlers in that outlier's population respectively.
populations = defaultdict(set)
indices = list(df.index.values)
i=[0]
def build_populations(outlier_id,index):
    populations[outlier_id].add(indices[i[0]])
    i[0]+=1

df.apply(lambda row: build_populations(row['outlier_id'],row), axis=1)
x=0
final_pop=defaultdict(set)
for p in populations:
    if len(populations[p])>1:
        final_pop[p] = populations[p]


def update_score(index,frac):
    df.at[index, 'severity_score'] = df.loc[index]['severity_score']*frac

# The following two functions are to be called when we find a bug/false-positive. It takes the index of the outlier as the argument.
def found_bug(idx,df = df):
    o_id = df.loc[idx]['outlier_id']
    if o_id in final_pop:
        l = final_pop[o_id]
        for n in l:
            update_score(n,1.1)



def found_fp(idx,df=df):
    o_id = df.loc[idx]['outlier_id']
    if o_id in final_pop:
        l = final_pop[o_id]
        for n in l:
            update_score(n,0.9)



## The following assigns severity to the outliers using Kmeans, 

kmeans = KMeans(n_clusters=4)
df['label'] = kmeans.fit_predict(df[['severity_score']])
centers = []
for i in kmeans.cluster_centers_:
    centers.append(i[0])
centers_index = np.argsort(centers)[::-1]
label_dict = {}
for i,it in enumerate(centers_index):
    label_dict[it]=i
final_ranks = []
for i in centers:
    final_ranks.append(list(df[df['label']==i].index))
def assign_label(label):
    return 4 - label_dict[label]
df['final_label'] = df.apply(lambda row: assign_label(row['label']), axis=1)
plt.rcParams['figure.figsize']=(15,10)
sns.scatterplot(x="Metric_1", y="Metric_2", size=df.index,sizes = (50,200),hue ="final_label",data=df)
plt.show()
sns.distplot(df['Metric_1'],kde = True)
plt.ylabel("Frequency")
plt.show()
sns.distplot(df['Metric_2'],kde = True)
plt.show()

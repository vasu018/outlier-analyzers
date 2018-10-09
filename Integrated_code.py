import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
from scipy.spatial import distance
style.use('ggplot')
from sklearn.cluster import KMeans

import pandas as pd
from sklearn.preprocessing import LabelBinarizer, MultiLabelBinarizer

#This is the intercluster distance criteria.
#In this criteria, the minimum distance between the centroids is used as the parameter.
#Optimal value for the weight has to be set.

def read_values_inter_cluster_criteria(main_list):
    debug_flag = 0
    l = []
    dimensions = len(main_list)
    for i in range(len(main_list[0])):
        temp = []
        for j in range(dimensions):
            temp.append(main_list[j][i])
        l.append(temp)
    if(debug_flag == 1):
        print("list of properties is")
        print(l)

    no_clusters = 2
    clf = KMeans(n_clusters = no_clusters)
    clf.fit(l)
    centroids = clf.cluster_centers_
    if(debug_flag == 1):
        print(" Centroids are")
        print(centroids)

    labels = clf.labels_

    if(debug_flag == 1):
        for i in range(len(l)):
            print("coordinate:", l[i], "label:", labels[i], "centroid:", centroids[labels[i]])

    weight = 0.8
    if(debug_flag == 1):
        print("weight is")
        print(weight)
    cluster_distances = []
    for i in range(len(centroids) ):
        j = i + 1
        while(j < len (centroids)):
            cluster_distances.append(distance.euclidean(centroids[i], centroids[j]))
            j = j + 1
    if(debug_flag == 1):
        print("distance between the various clusters is as follows:")
        print(cluster_distances)
        print("minimum inter-cluster distance is")
    min_intercluster_dist = min(cluster_distances)
    if(debug_flag == 1):
        print("minimum distance between the clsuters is")
        print(min_intercluster_dist)
    #weighing parameter
    w = weight
    outliers1 = []
    for i in range(len(l)):
        if(distance.euclidean(l[i], centroids[labels[i]]) > min_intercluster_dist*w ):
            if(debug_flag == 1):
                print("outlier detected at index:", i)
                print("encoded outlier is", l[i])
            outliers1.append(i)


    if(debug_flag == 1):
        print("outliers by inter cluster criteria are ")
        print(outliers1)

    return outliers1

#This is the intracluster distance criteria.
# In this criteria, the minimum distance between the centroid and the own cluster elements is used as the parameter
# Optimal value for the threshold has to be set.

def read_values_intra_cluster_criteria(main_list):
    l = []
    debug_flag = 0
    dimensions = len(main_list)
    for i in range(len(main_list[0])):
        temp = []
        for j in range(dimensions):
            temp.append(main_list[j][i])
        l.append(temp)
    print(l)

    no_clusters = 2
    clf = KMeans(n_clusters=no_clusters)
    clf.fit(l)
    centroids = clf.cluster_centers_
    if(debug_flag == 1):
        print(" Centroids are")
        print(centroids)
    labels = clf.labels_
    if(debug_flag == 1):
        for i in range(len(l)):
            print("coordinate:", l[i], "label:", labels[i], "centroid:", centroids[labels[i]])

    threshold = 0.05
    if(debug_flag == 1):
        print("threshold is")
        print(threshold)
    points_cluster_dist= []
    for i in range(no_clusters):
        points_cluster_dist.append([])
    for i in range(len(l)):
        points_cluster_dist[labels[i]].append( distance.euclidean(l[i], centroids[labels[i]]) )

    outliers2=[]
    for i in range(len(l)):
        mini = min(points_cluster_dist[labels[i]])
        center_dist = distance.euclidean(l[i], centroids[labels[i]])
        if(mini < threshold *center_dist ):
            if(debug_flag == 1):
                print("outlier detected at index:", i)
                print("encoded outlier is", l[i])
            outliers2.append(i)
    if(debug_flag == 1):
        print("outliers by intra-cluster criteria are")
        print(outliers2)
    return outliers2













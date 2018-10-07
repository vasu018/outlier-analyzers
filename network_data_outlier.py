import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
from scipy.spatial import distance
style.use('ggplot')
from sklearn.cluster import KMeans

import pandas as pd
from sklearn.preprocessing import LabelBinarizer, MultiLabelBinarizer



multiclass_feature_Xi = [("1.1.1.1", "2.2.2.2"),
                      ("1.1.1.1", "3.3.3.3"),
                      ("3.3.3.3", ""),
                      ("1.1.1.1", ""),
                      ("1.1.1.1", "2.2.2.2"),
                      ("1.1.1.1", "2.2.2.2"),
                      ("1.1.1.1", "2.2.2.2"),
                      ("1.1.1.1", "2.2.2.2"),
                      ("1.1.1.1", "3.3.3.3"),
                      ("1.1.1.1", "3.3.3.3"),
                      ("1.1.1.1", ""),
                      ("1.1.1.1", "3.3.3.3"),
                      (),
                      ("", ""),
                      ("1.1.1.1", ""),
                      ("2.2.2.2", "")]


NTP_Servers = [("1.1.1.1", "2.2.2.2"),
                      ("1.1.1.1", "3.3.3.3"),
                      ("3.3.3.3", ""),
                      ("1.1.1.1", ""),
                      ("1.1.1.1", "2.2.2.2"),
                      ("1.1.1.1", "2.2.2.2"),
                      ("1.1.1.1", "2.2.2.2"),
                      ("1.1.1.1", "2.2.2.2"),
                      ("1.1.1.1", "3.3.3.3"),
                      ("1.1.1.1", "3.3.3.3"),
                      ("1.1.1.1", ""),
                      ("1.1.1.1", "3.3.3.3"),
                      (),
                      ("", ""),
                      ("1.1.1.1", ""),
                      ("2.2.2.2", "")]



one_hot_multiclass = MultiLabelBinarizer()
multiClassEncodedList = one_hot_multiclass.fit_transform(multiclass_feature_Xi)

print("# Multi-class encoded features:\n", multiClassEncodedList)
print()

no_clusters = 2
clf = KMeans(n_clusters = no_clusters)
clf.fit(multiClassEncodedList)
centroids = clf.cluster_centers_
print(" Centroids are")
print(centroids)

labels = clf.labels_
for i in range(len(multiClassEncodedList)):
    print("coordinate:", multiClassEncodedList[i], "label:", labels[i], "centroid:", centroids[labels[i]])
    
#This is the intercluster distance criteria. In this criteria, the minimum distance between the centroids is used as the parameter and optimal value for the weight has to be set.
def criteria1(weight,labels,l):
    print("weight is")
    print(weight)
    cluster_distances = []
    for i in range(len(centroids) ):
        j = i + 1
        while(j < len (centroids)):
            cluster_distances.append(distance.euclidean(centroids[i], centroids[j]))
            j = j + 1

    print(cluster_distances)
    print("minimum inter-cluster distance is")
    min_intercluster_dist = min(cluster_distances)
    print(min_intercluster_dist)
    #weighing parameter
    w = weight
    outliers = []
    for i in range(len(l)):
        if(distance.euclidean(l[i], centroids[labels[i]]) > min_intercluster_dist*w ):
            outliers.append(l[i])
            print("outlier detected at index:", i)
            print("encoded outlier is", l[i])



          
    print("total number of outliers is ")
    print(len(outliers))


#This is the intracluster distance criteria. In this criteria, the minimum distance between the centroid and the own cluster elements is used as the parameter and optimal value for the threshols has to be set.
def criteria2(t,labels,l):
    print("Criteria 2")
    print("threshold is")
    print(t)
    points_cluster_dist= []
    for i in range(no_clusters):
        points_cluster_dist.append([])
    for i in range(len(l)):
        points_cluster_dist[labels[i]].append( distance.euclidean(l[i], centroids[labels[i]]) )

    threshold = t
    outliers2=[]
    for i in range(len(l)):
        mini = min(points_cluster_dist[labels[i]])
        center_dist = distance.euclidean(l[i], centroids[labels[i]])
        if(mini < threshold *center_dist ):
            outliers2.append(l[i])
            print("outlier detected at index:", i)
            print("encoded outlier is", l[i])
            
    print("total number of outliers is ")
    print(len(outliers2))


criteria1(0.8,labels,multiClassEncodedList)



criteria2(0.05, labels, multiClassEncodedList)










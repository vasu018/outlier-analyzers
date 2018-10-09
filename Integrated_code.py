import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
from scipy.spatial import distance
style.use('ggplot')
from sklearn.cluster import KMeans

import pandas as pd
from sklearn.preprocessing import LabelBinarizer, MultiLabelBinarizer


def read_values_inter_cluster_criteria(main_list):
    l = []
    dimensions = len(main_list)
    for i in range(len(main_list[0])):
        temp = []
        for j in range(dimensions):

            print("appending", main_list[j][i])
            temp.append(main_list[j][i])
        l.append(temp)
        #l.append([main_list[0][i],main_list[1][i]])
    print(l)



    no_clusters = 2
    clf = KMeans(n_clusters = no_clusters)
    clf.fit(l)
    centroids = clf.cluster_centers_
    print(" Centroids are")
    print(centroids)

    labels = clf.labels_

    for i in range(len(l)):
        print("coordinate:", l[i], "label:", labels[i], "centroid:", centroids[labels[i]])
        #plt.plot(l[i][0], l[i][1], colors[labels[i]], markersize = 20)
    #plt.scatter(centroids[:,0], centroids[:,1], marker = 'x', s = 150, linewidths=5, zorder=10)
    #plt.show()


#print("distance is")
#print(distance.euclidean(centroids[0], centroids[1]))
    weight = 0.8
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
    outliers1 = []
    for i in range(len(l)):
        if(distance.euclidean(l[i], centroids[labels[i]]) > min_intercluster_dist*w ):
            #outliers1.append(l[i])
            print("outlier detected at index:", i)
            outliers1.append(i)
            print("encoded outlier is", l[i])



            #plt.plot(l[i][0], l[i][1], colors[labels[0]], markersize=50)
    #plt.scatter(centroids[:,0], centroids[:,1], marker = 'x', s = 150, linewidths=5, zorder=10)

    #plt.show()
    #print("outliers by criterai 1 are")
    #print(outliers)
    print("outliers by inter cluster criteria are ")
    print(outliers1)

    return outliers1

# criteria 2
def read_values_intra_cluster_criteria(main_list):
    l = []
    dimensions = len(main_list)
    for i in range(len(main_list[0])):
        temp = []
        for j in range(dimensions):
            print("appending", main_list[j][i])
            temp.append(main_list[j][i])
        l.append(temp)
        # l.append([main_list[0][i],main_list[1][i]])
    print(l)

    no_clusters = 2
    clf = KMeans(n_clusters=no_clusters)
    clf.fit(l)
    centroids = clf.cluster_centers_
    print(" Centroids are")
    print(centroids)

    labels = clf.labels_

    for i in range(len(l)):
        print("coordinate:", l[i], "label:", labels[i], "centroid:", centroids[labels[i]])
        # plt.plot(l[i][0], l[i][1], colors[labels[i]], markersize = 20)
    # plt.scatter(centroids[:,0], centroids[:,1], marker = 'x', s = 150, linewidths=5, zorder=10)
    # plt.show()

# print("distance is")
# print(distance.euclidean(centroids[0], centroids[1]))
         
    threshold = 0.05
    print("Criteria 2")
    print("threshold is")
    print(threshold)
    points_cluster_dist= []
    for i in range(no_clusters):
        points_cluster_dist.append([])
    for i in range(len(l)):
        points_cluster_dist[labels[i]].append( distance.euclidean(l[i], centroids[labels[i]]) )
#print("new list is")
#print(points_cluster_dist)
#print("length of new is")
#print(len(points_cluster_dist))
#print("labels is")
#print(labels)

    outliers2=[]
    for i in range(len(l)):
        mini = min(points_cluster_dist[labels[i]])
        center_dist = distance.euclidean(l[i], centroids[labels[i]])
        if(mini < threshold *center_dist ):
            #outliers2.append(l[i])
            print("outlier detected at index:", i)
            print("encoded outlier is", l[i])
            outliers2.append(i)
            #plt.plot(l[i][0], l[i][1], colors[labels[0]], markersize=50)
    #plt.scatter(centroids[:,0], centroids[:,1], marker = 'x', s = 150, linewidths=5, zorder=10)
    #plt.show()
    #print("outliers are")
    #print(outliers2)
    print("outliers by intra-cluster criteria are")
    print(outliers2)
    return outliers2
#plt.show()












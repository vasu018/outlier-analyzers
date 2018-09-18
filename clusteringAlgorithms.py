import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
from scipy.spatial import distance
style.use('ggplot')
from sklearn.cluster import KMeans
#X = np.array([[5,6], [1,2], [0,1], [6,6], [7,11], [0,2]])
#plt.scatter(X[:,0], X[:,1], s = 150, linewidths = 5)
#plt.show()

#N = 4

def read_values():
    l = []
    for i in range(100):
        l.append(list(np.random.randint(100, size=2)))
    print(l)
    return l
l = read_values()

no_clusters = 2
clf = KMeans(n_clusters = no_clusters)
clf.fit(l)
centroids = clf.cluster_centers_
print("hey centroids are")
print(centroids)

labels = clf.labels_
colors = 5*["g.", "c.", "b.", "r.", "k." ]
for i in range(len(l)):
    print("coordinate:", l[i], "label:", labels[i], "centroid:", centroids[labels[i]])
    plt.plot(l[i][0], l[i][1], colors[labels[i]], markersize = 20)
#plt.scatter(centroids[:,0], centroids[:,1], marker = 'x', s = 150, linewidths=5, zorder=10)
#plt.show()

a = (1, 2, 3)
b = (4, 5, 6)
print("distance is")
print(distance.euclidean(centroids[0], centroids[1]))
def criteria1(weight,labels,l):

    cluster_distances = []
    for i in range(len(centroids) ):
        j = i + 1
        while(j < len (centroids)):
            cluster_distances.append(distance.euclidean(centroids[i], centroids[j]))
            j = j + 1

    print(cluster_distances)
    print("minimum inter-cluster distance is")
    min_intercluster_dist = min(cluster_distances)

    #weighing parameter
    w = weight
    outliers = []
    for i in range(len(l)):
        if(distance.euclidean(l[i], centroids[labels[i]]) > min_intercluster_dist*w ):
            outliers.append(l[i])
            plt.plot(l[i][0], l[i][1], colors[labels[0]], markersize=50)
    plt.scatter(centroids[:,0], centroids[:,1], marker = 'x', s = 150, linewidths=5, zorder=10)

    plt.show()
    print("outliers by criterai 1 are")
    print(outliers)
    print("total number of outliers is ")
    print(len(outliers))


# criteria 2
def criteria2(t,labels,l):
    print("new criteria 2")
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
    threshold = t
    outliers2=[]
    for i in range(len(l)):
        mini = min(points_cluster_dist[labels[i]])
        center_dist = distance.euclidean(l[i], centroids[labels[i]])
        if(mini < threshold *center_dist ):
            outliers2.append(l[i])
        #plt.plot(l[i][0], l[i][1], colors[labels[0]], markersize=50)
#plt.scatter(centroids[:,0], centroids[:,1], marker = 'x', s = 150, linewidths=5, zorder=10)
    print("outliers are")
    print(outliers2)
    print("total number of outliers is ")
    print(len(outliers2))
#plt.show()
criteria1(1,labels,l)
criteria2(0.001, labels, l)







import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
from scipy.spatial import distance
style.use('ggplot')
from sklearn.cluster import KMeans

import pandas as pd
import random
from math import log
import math
import copy

# Importing pybatfish APIs.
from pybatfish.client.commands import *
from pybatfish.question.question import load_questions, list_questions
from pybatfish.question import bfq

from sklearn.preprocessing import LabelBinarizer, MultiLabelBinarizer

class Outlier():

    def __init__(self, question, prop):
                
        load_questions()
        bf_init_snapshot('datasets/networks/example')

        command = "self.result = " + question
        exec(command)

        
        data = list(self.result[prop])
        self.multiclass_feature = []
        for d in data:
            self.multiclass_feature.append(tuple(d))

        # self.multiclass_feature = [("1.1.1.1", "2.2.2.2", "1500"),
        #                       ("1.1.1.1", "3.3.3.3", "1500"),
        #                       ("3.3.3.3", "", "9100"),
        #                       ("1.1.1.1",""),
        #                       ("1.1.1.1", "2.2.2.2", "1500"),
        #                       ("1.1.1.1", "2.2.2.2", "1500"),
        #                       ("1.1.1.1", "2.2.2.2", "1500"),
        #                       ("1.1.1.1", "2.2.2.2", "1500"),
        #                       ("1.1.1.1", "3.3.3.3", "9100"),
        #                       ("1.1.1.1", "3.3.3.3", "9100"),
        #                       ("1.1.1.1", "", "1500"),
        #                       ("1.1.1.1", "2.2.2.2", "1500"),
        #                       (),
        #                       ("", "", ""),
        #                       ("1.1.1.1", "", ""),
        #                       ("2.2.2.2","9100")]


    def encode(self):
        # Create multiclass multilabelbinarizer
        one_hot_multiclass = MultiLabelBinarizer()
        multiClassEncodedList = one_hot_multiclass.fit_transform(self.multiclass_feature)


        print("# Multi-class encoded features:\n", multiClassEncodedList)
        print()

        uniqueClasses = one_hot_multiclass.classes_

        print("# Unique Classes:", uniqueClasses)
        print()

        '''
        Count the frequency of specific attribute value in the complete data that is 
        constructed using the  MultiLabelBinarizer(). 
        '''
        mClassElementCountList = []
        uniqueClassesNonNull = []
        for counter, mClass in enumerate(one_hot_multiclass.classes_):
            uniqueClassesNonNull.append(mClass)
            mClassDensityValue = 0
            for multiClassElementCode in multiClassEncodedList:
                mClassDensityValue = mClassDensityValue + multiClassElementCode[counter]
            mClassElementCountList.append(mClassDensityValue)

        print("# Each class count:", mClassElementCountList)
        print()

        ''' 
        Calculate the density of each attribute value in the data entries (Xi).
        '''
        entryDensityListNormalize = []
        self.entryDensityList = []
        for classCounter, entryVector in enumerate(multiClassEncodedList):
            overallProportion = 1 / (len(uniqueClassesNonNull) * len(self.multiclass_feature))
            summationVectorVals = 0
            for entryCounter, entryVectorClassValue in enumerate(entryVector):
                if (uniqueClasses[entryCounter] == ''):
                    continue
                summationVectorVals = summationVectorVals + (entryVectorClassValue * mClassElementCountList[entryCounter])
            densityXi = overallProportion * summationVectorVals
            entryDensityListNormalize.append(densityXi)
            self.entryDensityList.append(summationVectorVals)
        print(self.entryDensityList)

    def tukey(self):

        # nums = np.array(nums)

        q1 = np.percentile(self.entryDensityList, 25)
        q3 = np.percentile(self.entryDensityList, 75)

        iqr = q3 - q1

        lower_distance = q1 - 1.5 * iqr
        upper_distance = q3 + 1.5 * iqr

        outliers = []

        for n in self.entryDensityList:
            if n < lower_distance or n > upper_distance:
                outliers.append(n)

        return outliers


    def z_score(self):
        # nums should be a list of integers

        # nums = np.array(nums)

        mean = np.mean(self.entryDensityList)
        std = np.std(self.entryDensityList)

        lower_boundary = mean - std * 3
        upper_boundary = mean + std * 3

        outliers = []

        for n in self.entryDensityList:
            if n <= lower_boundary or n >= upper_boundary:
                outliers.append(n)

        return outliers


    def modified_z_score(self):
        # nums should be a list of integers

        median = np.median(self.entryDensityList)

        df = pd.DataFrame()
        df['a'] = self.entryDensityList
        mad = df['a'].mad()

        lower_boundary = median - mad * 3
        upper_boundary = median + mad * 3

        outliers = []

        for n in self.entryDensityList:
            if n <= lower_boundary or n >= upper_boundary:
                outliers.append(n)

        return outliers



out = Outlier("bfq.nodeProperties().answer().frame()", "NTP_Servers")
# out = Outlier("bfq.interfaceProperties().answer().frame()", "Interface_Type")
out.encode()
# print(out.tukey())
# print(out.z_score())
# print(out.modified_z_score())





def log_normalize(nums):
    return [log(n) for n in nums]



def regression(points):
    # pointers should be a list of pairs of numbers (tuples)

    n = len(points)

    sum_x = 0.0
    sum_y = 0.0
    sum_xy = 0.0
    sum_x2 = 0.0
    sum_y2 = 0.0


    for i in range(n):

        x = points[i][0]
        y = points[i][1]

        sum_x += x
        sum_y += y
        sum_xy += x * y
        sum_x2 += x * x
        sum_y2 += y * y

    a = (sum_y * sum_x2 - sum_x * sum_xy) / (n * sum_x2 - sum_x * sum_x)

    b = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)

    return a, b





def generate_list():
    nums = []
    for _ in range(100):
        nums.append(random.randint(1, 20))

    for _ in range(5):
        nums.append(random.randint(20, 100))

    nums.sort()
    return nums


def generate_point():
    x = random.randint(1, 50)
    y = x * (random.random() * 2 + 1)
    # y = random.randint(1,200)
    y = int(y)
    return (x,y)


def generate_points():
    points = []

    for _ in range(100):
        points.append(generate_point())

    points.append((1000,1000))

    points.sort()

    return points

def predict(x, a, b):
    y = a + b * x
    return y

def mean_squared_error(points, a, b):

    mse = 0



    for i in range(len(points)):
        prediction = predict(points[i][0], a, b)
        error = prediction - points[i][1]
        mse += error * error

    mse /= len(points)

    return mse


def cooks_distance(points):
    # points should be a list of pairs of numbers (tuples)

    a, b = regression(points)

    outliers = []

    s = mean_squared_error(points, a, b)

    for i in range(len(points)):
        points_missing = copy.deepcopy(points)
        del points_missing[i]

        a_missing, b_missing = regression(points_missing)

        distance = 0

        # print(predict(points[i][0], a, b) - predict(points[i][0], a_missing, b_missing))

        for j in range(len(points)):
            distance += math.pow((predict(points[i][0], a, b) - predict(points[i][0], a_missing, b_missing)), 2)

        distance /= (3 * s)

        print(distance)

        if distance > 1:
            outliers.append(points[i])

    return outliers



#def read_values():
#    l = []
#    for i in range(100):
#        l.append(list(np.random.randint(100, size=2)))
#    print(l)
#    return l
#l = read_values()

#no_clusters = 2
#clf = KMeans(n_clusters = no_clusters)
#clf.fit(l)
#centroids = clf.cluster_centers_
#print("hey centroids are")
#print(centroids)

#labels = clf.labels_
#colors = 5*["g.", "c.", "b.", "r.", "k." ]
#for i in range(len(l)):
#    print("coordinate:", l[i], "label:", labels[i], "centroid:", centroids[labels[i]])
#    plt.plot(l[i][0], l[i][1], colors[labels[i]], markersize = 20)

#a = (1, 2, 3)
#b = (4, 5, 6)
#print("distance is")
#print(distance.euclidean(centroids[0], centroids[1]))
#def criteria1(weight,labels,l):

#    cluster_distances = []
#    for i in range(len(centroids) ):
#        j = i + 1
#        while(j < len (centroids)):
#            cluster_distances.append(distance.euclidean(centroids[i], centroids[j]))
#            j = j + 1

#    print(cluster_distances)
#    print("minimum inter-cluster distance is")
#    min_intercluster_dist = min(cluster_distances)

#    #weighing parameter
#    w = weight
#    outliers = []
#    for i in range(len(l)):
#        if(distance.euclidean(l[i], centroids[labels[i]]) > min_intercluster_dist*w ):
#            outliers.append(l[i])
#            plt.plot(l[i][0], l[i][1], colors[labels[0]], markersize=50)
#    plt.scatter(centroids[:,0], centroids[:,1], marker = 'x', s = 150, linewidths=5, zorder=10)

#    plt.show()
#    print("outliers by criterai 1 are")
#    print(outliers)
#    print("total number of outliers is ")
#    print(len(outliers))


## criteria 2
#def criteria2(t,labels,l):
#    print("new criteria 2")
#    points_cluster_dist= []
#    for i in range(no_clusters):
#        points_cluster_dist.append([])
#    for i in range(len(l)):
#        points_cluster_dist[labels[i]].append( distance.euclidean(l[i], centroids[labels[i]]) )
#    threshold = t
#    outliers2=[]
#    for i in range(len(l)):
#        mini = min(points_cluster_dist[labels[i]])
#        center_dist = distance.euclidean(l[i], centroids[labels[i]])
#        if(mini < threshold *center_dist ):
#            outliers2.append(l[i])
#    print("outliers are")
#    print(outliers2)
#    print("total number of outliers is ")
#    print(len(outliers2))
#criteria1(1,labels,l)
#criteria2(0.001, labels, l)







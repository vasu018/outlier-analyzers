import numpy as np
import pandas as pd
import copy
import math
from sklearn.cluster import KMeans
from scipy.spatial import distance
from sklearn import mixture
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest


def test(densityList):
    print(densityList)
    for d in densityList:
        print(d)

#TODO
#Correct z_score

def tukey(densityList):

    q1 = np.percentile(densityList, 25)
    q3 = np.percentile(densityList, 75)

    iqr = q3 - q1

    lower_distance = q1 - 1.5 * iqr
    upper_distance = q3 + 1.5 * iqr

    outliers = []

    for i, n in enumerate(densityList):
        if n < lower_distance or n > upper_distance:
            outliers.append(i)
    return outliers


def z_score(densityList):
    mean = np.mean(densityList)
    std = np.std(densityList)

    outliers = []

    for i, n in enumerate(densityList):
        z = (n - mean) / std
        if abs(z) >= 1:
            outliers.append(i)
    return outliers


def modified_z_score(densityList):
    median = np.median(densityList)
    df = pd.DataFrame()
    df['a'] = densityList
    mad = df['a'].mad()

    outliers = []

    for i, n in enumerate(densityList):
        z = (n - median) / mad
        if abs(z) >= 1:
            outliers.append(i)

    return outliers


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

        # print(distance)

        if distance > 1:
            # outliers.append(points[i])
            outliers.append(i)

    return outliers


def mahalanobis_distance(densityLists):

    vectors = []
    for i in range(len(densityLists[0])):
        vector = []

        for j in range(len(densityLists)):
            vector.append(densityLists[j][i]) 
        vectors.append(vector) 

    # calculate average vector
    average_vector = [0] * len(densityLists)
    for vector in vectors:
       for i in range(len(vector)):
           average_vector[i] += vector[i]

    for i in range(len(average_vector)):
        average_vector[i] /= len(vectors)

    # calculate mahalanobis distance for each point
    outliers = []

    for i, vector in enumerate(vectors):
        combination = np.vstack((vector, average_vector))
        covariance_matrix = np.cov(combination)
        mahalanobis_dist = distance.mahalanobis(vector, average_vector, covariance_matrix)

        if mahalanobis_dist > 500:
            outliers.append(i)


    return outliers
        

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

    weight = 0.1
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


def Gaussian(encodedLists):
    #Gaussian Mixture is used for soft clustering. Insted of assigning points to specific classes it assigns probability.
    #The n_components parameter in the Gaussian is used to specify the number of Gaussians.
    concatenated_features = []
    for i in range(len(encodedLists[0])):
        temp = []
        for j in range(len(encodedLists)):
            temp.extend(encodedLists[j][i])    
        concatenated_features.append(temp)   
    
    print("concateanted feature is")
    print(concatenated_features)
    clf = mixture.GaussianMixture(n_components=2, covariance_type='full')
    clf.fit(concatenated_features)
    Z = -clf.score_samples(np.array(concatenated_features))
    return Z


def KNN(encodedLists):
    concatenated_features = []
    for i in range(len(encodedLists[0])):
        temp = []
        for j in range(len(encodedLists)):
            temp.extend(encodedLists[j][i])    
        concatenated_features.append(temp)   
    
    print("concateanted feature is")
    print(concatenated_features)
    nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(concatenated_features)
    distances, indices = nbrs.kneighbors(concatenated_features)
    print("indices in KNN are")
    print(indices)
    print("distances in KNN are")
    print(distances)


def severity(density_list):
    # Severity between features is calculated. To calculate severity we need to pass density lists of features. 
    # Currently, we are calculating severity based on correlation coefficients.  
    # Correlation coefficient gives how closely two features are linked to each other.
    print("In severity function:")
    for feature1 in range(len(density_list) - 1):
        feature2 = feature1 + 1
        while(feature2<len(density_list)):
            print("correlation between features indexed" ,feature1)
            print("and")
            print(feature2)
            print(np.corrcoef(density_list[feature1], density_list[feature2])[0, 1])
            feature2 = feature2 + 1


def RandomForests(densityList,encodedLists):
    #First apply an existing outlier detection technique as RandomForests works on supervised data.

    mean = np.mean(densityList)
    std = np.std(densityList)

    outliers = []
    labels = []
    print("In RandomForests method")
    print("density list is", densityList)

    for i, n in enumerate(densityList):
        z = (n - mean) / std
        if abs(z) >= 1:
            outliers.append(i)
            labels.append(1)
        else:
            labels.append(0)
    print("labels are", labels)

    concatenated_features = []
    for i in range(len(encodedLists[0])):
        temp = []
        for j in range(len(encodedLists)):
            temp.extend(encodedLists[j][i])    
        concatenated_features.append(temp)   
    
    print("concateanted feature is")
    print(concatenated_features)

    X_train, X_test, y_train, y_test = train_test_split(concatenated_features, labels, test_size=0.33, random_state=42)
    
    clf = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)


    clf.fit(X_train, y_train)
    print("RandomForests predictions are")
    print(clf.predict(X_test))
    print("Actual classification is")
    print(y_test)


def isolationForests(densityList,encodedLists):
    #First apply an existing outlier detection technique as RandomForests works on supervised data.

    mean = np.mean(densityList)
    std = np.std(densityList)

    outliers = []
    labels = []
    print("In RandomForests method")
    print("density list is", densityList)

    for i, n in enumerate(densityList):
        z = (n - mean) / std
        if abs(z) >= 1:
            outliers.append(i)
            labels.append(1)
        else:
            labels.append(0)
    print("labels are", labels)

    concatenated_features = []
    for i in range(len(encodedLists[0])):
        temp = []
        for j in range(len(encodedLists)):
            temp.extend(encodedLists[j][i])    
        concatenated_features.append(temp)   
    
    print("concateanted feature is")
    print(concatenated_features)
    X_train, X_test, y_train, y_test = train_test_split(concatenated_features, labels, test_size=0.33, random_state=42)
    clf = IsolationForest(max_samples=100)
                      
    clf.fit(X_train)
    y_pred_train = clf.predict(X_train)
    y_pred_test = clf.predict(X_test)
    print("isolationForests predictions on train data are")
    print(y_pred_train)
    print("isolationForests predictions on test data are")
    print(y_pred_test)


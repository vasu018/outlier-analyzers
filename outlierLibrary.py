import numpy as np
import pandas as pd
import copy
import math

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

from scipy.spatial import distance


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
        



# Import required packages
# from sklearn import preprocessing
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import LabelBinarizer, MultiLabelBinarizer

# importing Statistics module
import statistics
import numpy

# # Create feature
# feature = np.array([["Texas"],
#                     ["California"],
#                     ["Texas"],
#                     ["Delaware"],
#                     ["Texas"]])
#
# # Create one-hot encoder
# one_hot = LabelBinarizer()
#
# #
# # One-hot encode feature
# one_hot.fit_transform(feature)
# #
#
# # Reverse one-hot encoding
# ret = one_hot.inverse_transform(one_hot.transform(feature))
# print(ret)
# print(one_hot.classes_)
#
# # Create dummy variables from feature
# ret = pd.get_dummies(feature[:,0])
# print("One-class encoding:\n", ret)

#######

# raw_data = {'patient': [1, 1, 1, 2, 2],
#         'obs': [1, 2, 3, 1, 2],
#         'treatment': [0, 1, 0, 1, 0],
#         'score': ['strong', 'weak', 'normal', 'weak', 'strong']}
# df = pd.DataFrame(raw_data, columns = ['patient', 'obs', 'treatment', 'score'])
# print(df)
#
# # Create a label (category) encoder object
# le = preprocessing.LabelEncoder()
#
# # Fit the encoder to the pandas column
# le.fit(df['score'])
# # print(df)
# print("######################")


#
# Create multiclass feature
#
multiclass_feature = [("1.1.1.1", "2.2.2.2", "1500"),
                      ("1.1.1.1", "3.3.3.3", "1500"),
                      ("3.3.3.3", "", "9100"),
                      ("1.1.1.1",""),
                      ("1.1.1.1", "2.2.2.2", "1500"),
                      ("1.1.1.1", "2.2.2.2", "1500"),
                      ("1.1.1.1", "2.2.2.2", "1500"),
                      ("1.1.1.1", "2.2.2.2", "1500"),
                      ("1.1.1.1", "3.3.3.3", "9100"),
                      ("1.1.1.1", "3.3.3.3", "9100"),
                      ("1.1.1.1", "", "1500"),
                      ("1.1.1.1", "2.2.2.2", "1500"),
                      (),
                      ("", "", ""),
                      ("1.1.1.1", "", ""),
                      ("2.2.2.2","9100")]

multiclass_feature2 = [("1.1.1.1", "2.2.2.2"),
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
                      ("2.2.2.2","")]

# Input Multi-class feature.
print("Multi-class feature:\n", multiclass_feature)

# Create multiclass one-hot encoder
one_hot_multiclass = MultiLabelBinarizer()

# One-hot encode multiclass feature
multiClassEncodedList = one_hot_multiclass.fit_transform(multiclass_feature)
print("Multi-class encoding:\n", multiClassEncodedList)

#
# Debug print messages
#
uniqueClasses = one_hot_multiclass.classes_
print("Unique Classes:", uniqueClasses)

# Calculate the count of each class
# i.e., count of column elements in the encoded multi-class array.
mClassElementCountList = []
uniqueClassesNonNull = []
for counter, mClass in enumerate(one_hot_multiclass.classes_):
    # if (uniqueClasses[counter] == ''):
    #     continue

    # print(mClass)
    uniqueClassesNonNull.append(mClass)
    mClassDensityValue = 0
    for multiClassElementCode in multiClassEncodedList:
        mClassDensityValue = mClassDensityValue + multiClassElementCode[counter]
    mClassElementCountList.append(mClassDensityValue)

print("Each class count:", mClassElementCountList)

# Calculate the density of each element in the data list Xi
entryDensityListNormalize = []
entryDensityList = []
for classCounter, entryVector in enumerate(multiClassEncodedList):
    overallProportion = 1 / (len(uniqueClassesNonNull) * len(multiclass_feature))
    summationVectorVals = 0
    for entryCounter, entryVectorClassValue in enumerate(entryVector):
        if (uniqueClasses[entryCounter] == ''):
            continue
        summationVectorVals = summationVectorVals + (entryVectorClassValue * mClassElementCountList[entryCounter])
        # print(entryCounter)
    densityXi = overallProportion * summationVectorVals
    entryDensityListNormalize.append(densityXi)
    entryDensityList.append(summationVectorVals)

print("Density Values list:\n", entryDensityListNormalize)
print("Density Values list (Non-normalize):\n", entryDensityList)


# Calculate Mean and Standard deviation on the density values of data set.
meanDataSet =  numpy.mean(entryDensityList)
medianDataSet = numpy.median(entryDensityList)
standardDeviationDataSet = statistics.stdev(entryDensityList)
print("# Mean:", meanDataSet)
print("# Standard Deviation:", standardDeviationDataSet)

# Calculate ZScores and Outliers.
outliersMean = []
outliersMedian = []

for entryDensityCounter, eachValue in enumerate(entryDensityList):
    zScoreMean = abs((eachValue - meanDataSet) / (standardDeviationDataSet))
    zScoreMedian = abs((eachValue - medianDataSet) / (standardDeviationDataSet))

    # Debug message
    # print("Actual Value:",eachValue, "zScore:", zScoreMean)
    print("Actual Value:",eachValue, "zScore:", zScoreMedian)

    if (zScoreMean >= 0.9):
        outliersMean.append(multiclass_feature[entryDensityCounter])

    if (zScoreMedian >= 0.9):
        outliersMedian.append(multiclass_feature[entryDensityCounter])


print("# Outliers with mean:")
print(outliersMean)

print("# Outliers with Median:")
print(outliersMedian)
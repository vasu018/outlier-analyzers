'''
Module to calculate the outliers for network configurations that is supplied
as features. Outliers are calculated on the multi-class features using the
zScore and modified zScore Outlier analyzer technique.

High level design:

Step 1: From the data frames generated as output of the question is converted to
multi-class feature.

Step 2: The multi-class is feature is encoded using the oneHot Encoding or
MultiLabel Binarizer.

Step 3: The encoded data is converted to some distribution using techniques such as:
    (a) rough clustering and
    (b) ...

Step 4: The distribution is supplied to different outlier Analyzer techniques for
Outlier detection:
    (a) zScore
    (b) modified zScore
    (c) ...


<Input>: question-type, question-parameters, custom-threshold
<Output>: Outliers
'''

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelBinarizer, MultiLabelBinarizer
import statistics
import numpy

DEBUG_PRINT_FLAG = False

'''
Example list of multiclass features.

[TODO]: These static features will be extracted from the pybatfish questions 
data frames.
'''

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

print("# Multi-class feature:\n", multiclass_feature)
print()

# Create multiclass multilabelbinarizer
one_hot_multiclass = MultiLabelBinarizer()
multiClassEncodedList = one_hot_multiclass.fit_transform(multiclass_feature)


print("# Multi-class encoded features:\n", multiClassEncodedList)
print()

uniqueClasses = one_hot_multiclass.classes_

if DEBUG_PRINT_FLAG:
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

if DEBUG_PRINT_FLAG:
    print("# Each class count:", mClassElementCountList)
    print()

''' 
Calculate the density of each attribute value in the data entries Xi.
'''
entryDensityListNormalize = []
entryDensityList = []
for classCounter, entryVector in enumerate(multiClassEncodedList):
    overallProportion = 1 / (len(uniqueClassesNonNull) * len(multiclass_feature))
    summationVectorVals = 0
    for entryCounter, entryVectorClassValue in enumerate(entryVector):
        if (uniqueClasses[entryCounter] == ''):
            continue
        summationVectorVals = summationVectorVals + (entryVectorClassValue * mClassElementCountList[entryCounter])
    densityXi = overallProportion * summationVectorVals
    entryDensityListNormalize.append(densityXi)
    entryDensityList.append(summationVectorVals)

if DEBUG_PRINT_FLAG:
    print("# Density Values list of data-set (Normalized):\n", entryDensityListNormalize)
    print()
    print("# Density Values list of data-set (Real):\n", entryDensityList)
    print()


''' 
Calculate Mean and Standard deviation on the density values of data set.
'''
meanDataSet =  numpy.mean(entryDensityList)
medianDataSet = numpy.median(entryDensityList)
standardDeviationDataSet = statistics.stdev(entryDensityList)

print("# Mean:", meanDataSet)
print("# Standard Deviation:", standardDeviationDataSet)

''' 
Calculate ZScores and Outliers.
'''
if DEBUG_PRINT_FLAG:
    print()
    print("## Actual Value \t | zScore (Mean) \t | zScore (Median)")

outliersMean = []
outliersMedian = []
for entryDensityCounter, eachValue in enumerate(entryDensityList):
    zScoreMean = abs((eachValue - meanDataSet) / (standardDeviationDataSet))
    zScoreMedian = abs((eachValue - medianDataSet) / (standardDeviationDataSet))

    if DEBUG_PRINT_FLAG:
        print("\t", eachValue,"\t\t\t\t", zScoreMean,"\t\t", zScoreMedian)

    # Currently one standard deviation is considered for calculating the outliers.
    if (zScoreMean >= 1):
        outliersMean.append(multiclass_feature[entryDensityCounter])

    if (zScoreMedian >= 1):
        outliersMedian.append(multiclass_feature[entryDensityCounter])



print("# Outliers with Mean on data-set:")
print(outliersMean)
print()
print("# Outliers with Median on data-set:")
print(outliersMedian)
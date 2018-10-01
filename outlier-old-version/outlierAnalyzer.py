"""
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
"""


import pandas as pd
from sklearn.preprocessing import LabelBinarizer, MultiLabelBinarizer
import statistics
import numpy
from collections import Counter
import sys
from colorama import Fore, Back, Style

# Importing pybatfish APIs.
from pybatfish.client.commands import *
from pybatfish.question.question import load_questions, list_questions
from pybatfish.question import bfq

# Debug Flag
DEBUG_PRINT_FLAG = True

# Static Threshold for comparison with density calculation
OUTLIER_THRESHOLD = 1.0/3.0

# Help flag
if len(sys.argv) < 3 or sys.argv[1] == '-h':
    print("#################################################################################")
    print(Fore.RED + "# Error: Invalid arguments !!!")
    print(Style.RESET_ALL)
    print(Fore.BLUE + "# Usage: python3 outlierAnalyzer.py <propertiesType> <propertyAttributes>")
    print("# Examples: python3 outlierAnalyzer.py nodeProperties NTP_Servers")
    print("#           python3 outlierAnalyzer.py interfaceProperties \"HSRP_GROUPS | MTU\"")
    print(Style.RESET_ALL)
    print("#################################################################################")
    sys.exit(0)

# Loading questions from pybatfish.
load_questions()

pd.compat.PY3 = True
PD_DEFAULT_COLWIDTH = 250
pd.set_option('max_colwidth', PD_DEFAULT_COLWIDTH)

bf_init_snapshot('datasets/networks/example')

'''
Example list of multiclass features.
'''

# multiclass_feature_Xi = [("1.1.1.1", "2.2.2.2", "1500"),
#                       ("1.1.1.1", "3.3.3.3", "1500"),
#                       ("3.3.3.3", "", "9100"),
#                       ("1.1.1.1", ""),
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
#                       ("2.2.2.2", "9100")]

# multiclass_feature_Xi2 = [("1.1.1.1", "2.2.2.2"),
#                       ("1.1.1.1", "3.3.3.3"),
#                       ("3.3.3.3", ""),
#                       ("1.1.1.1", ""),
#                       ("1.1.1.1", "2.2.2.2"),
#                       ("1.1.1.1", "2.2.2.2"),
#                       ("1.1.1.1", "2.2.2.2"),
#                       ("1.1.1.1", "2.2.2.2"),
#                       ("1.1.1.1", "3.3.3.3"),
#                       ("1.1.1.1", "3.3.3.3"),
#                       ("1.1.1.1", ""),
#                       ("1.1.1.1", "3.3.3.3"),
#                       (),
#                       ("", ""),
#                       ("1.1.1.1", ""),
#                       ("2.2.2.2", "")]



# Read the question and property from the command line and parse the returned data frame
command = "result = bfq." + sys.argv[1] + "().answer().frame()"
exec(command)

props = sys.argv[2].split('|')
for i in range(len(props)):
    props[i] = props[i].strip() 

datas = []
for prop in props:
    data = list(result[prop])
    datas.append(data)


multiclass_feature_Xi = []

for i in range(len(datas[0])):
    element = []
    for data in datas:
        if type(data[i]) is list:
            # if len(data[i]) == 0:
            #     element.append('')
            for d in data[i]: 
                element.append(d)
        else:
            element.append(data[i])
    element = tuple(element)
    multiclass_feature_Xi.append(element)


print("# Feature set input:\n", multiclass_feature_Xi)
print()


'''
Approach 1: Simple Threshold-based outlier detection with outlier threshold 1/3
'''

# Unique instance counter of the elements.
valueCounterOutCome = Counter(multiclass_feature_Xi)
if DEBUG_PRINT_FLAG:
    print("# Unique Instance Counter:", valueCounterOutCome)

mostCommonElement = valueCounterOutCome.most_common(1)
print("# Most common element from the input data:", mostCommonElement)

mostCommonElementSize = valueCounterOutCome.most_common()[0][1]
if DEBUG_PRINT_FLAG:
    print("# Most common element size:", mostCommonElementSize)


totalSizeOfmultiClassSet = len(multiclass_feature_Xi)
if DEBUG_PRINT_FLAG:
    print("# Overall size of input data-Set:", totalSizeOfmultiClassSet)

outlierThresholdValue = (totalSizeOfmultiClassSet - mostCommonElementSize) / totalSizeOfmultiClassSet
print("# Outlier threshold on data:", outlierThresholdValue)
print()


outliersThresholdAppraoch = []
if (OUTLIER_THRESHOLD > 0 and outlierThresholdValue < OUTLIER_THRESHOLD):
    for entryCounter, entryValue in enumerate(valueCounterOutCome.elements()):
        if (entryValue != valueCounterOutCome.most_common()[0][0]):
            print("Outlier:", entryValue)
            outliersThresholdAppraoch.append(entryValue)
            # [TODO]: Just simple code.
            # Required calculation can de done later.

'''
Approach 2: Outlier detection using attribute density.
'''

# Create multiclass multilabelbinarizer
one_hot_multiclass = MultiLabelBinarizer()
multiClassEncodedList = one_hot_multiclass.fit_transform(multiclass_feature_Xi)


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
Calculate the density of each attribute value in the data entries (Xi).
'''
entryDensityListNormalize = []
entryDensityList = []
for classCounter, entryVector in enumerate(multiClassEncodedList):
    overallProportion = 1 / (len(uniqueClassesNonNull) * len(multiclass_feature_Xi))
    summationVectorVals = 0
    for entryCounter, entryVectorClassValue in enumerate(entryVector):
        if (uniqueClasses[entryCounter] == ''):
            continue
        summationVectorVals = summationVectorVals + (entryVectorClassValue * mClassElementCountList[entryCounter])
    densityXi = overallProportion * summationVectorVals
    entryDensityListNormalize.append(densityXi)
    entryDensityList.append(summationVectorVals)

if DEBUG_PRINT_FLAG:
    print("# Density Values list of data-set (Normalized Xi):\n", entryDensityListNormalize)
    print()
    print("# Density Values list of data-set (Real Xi):\n", entryDensityList)
    print()


''' 
Calculate Mean and Standard deviation on the density values of data set.
'''
meanDataSet =  numpy.mean(entryDensityList)
medianDataSet = numpy.median(entryDensityList)
standardDeviationDataSet = statistics.stdev(entryDensityList)

print("# Mean:", meanDataSet)
print("# Standard Deviation:", standardDeviationDataSet)
print()

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
        outliersMean.append(multiclass_feature_Xi[entryDensityCounter])

    if (zScoreMedian >= 1):
        outliersMedian.append(multiclass_feature_Xi[entryDensityCounter])

print("#")
print("# Outliers using simple threshold of 1/3 on data-set:")
print("#")
print(outliersThresholdAppraoch)
print()

print("#")
print("# Outliers with Mean on data-set:")
print("#")
print(outliersMean)
print()

print("#")
print("# Outliers with Median on data-set:")
print("#")
print(outliersMedian)

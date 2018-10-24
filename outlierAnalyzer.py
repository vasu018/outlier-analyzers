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
from sklearn.preprocessing import MultiLabelBinarizer
import sys
from colorama import Fore, Back, Style
import outlierLibrary
from collections import Counter
import json
from pprint import pprint

# Importing pybatfish APIs. 

from pybatfish.client.commands import *
from pybatfish.question.question import load_questions, list_questions
from pybatfish.question import bfq

# Setup
load_questions()

pd.compat.PY3 = True
PD_DEFAULT_COLWIDTH = 250
pd.set_option('max_colwidth', PD_DEFAULT_COLWIDTH)

bf_init_snapshot('datasets/networks/example')

# Debug flags
DEBUG_PRINT_FLAG = False
STEP_TWO_FLAG = True

# Static Threshold for comparison with density calcuation
OUTLIER_THRESHOLD = 1.0 / 3.0

def error_msg(help_flag):
    print("#################################################################################")
    if (not help_flag):
        print(Fore.RED + "# Error: Invalid arguments !!!")
        print(Style.RESET_ALL)
    print(Fore.BLUE + "# Usage: python3 outlierAnalyzer.py -t <propertiesType> -p <propertyAttributes>")
    print("# Examples: python3 outlierAnalyzer.py -t \"nodeProperties()\" -p NTP_Servers")
    print("#           python3 outlierAnalyzer.py -t \"interfaceProperties()\" -p \"HSRP_GROUPS | MTU\"")
    print(Fore.GREEN + "# Usage: python3 outlierAnalyzer.py -i <inputFile>")
    print("#           python3 outlierAnalyzer.py -i input.txt")
    print(Style.RESET_ALL)
    print("#################################################################################")
    sys.exit(0)


# Check if user input is correct
try:
    if sys.argv[1] == '-h':
        error_msg(True)
    elif sys.argv[1] == '-t' and sys.argv[3] == '-p' and sys.argv[5] == '-s':
        if len(sys.argv) != 7:
            error_msg(False)
        else:
            READ_FILE_FLAG = False    
    elif sys.argv[1] == '-i':
        if len(sys.argv) != 3:
            error_msg(False)
        else:
            READ_FILE_FLAG = True
    else:
        error_msg(False)
except:
    error_msg(False)


# utility functions

def listify(frame):
    outputList = list(frame)
    for i in range(len(outputList)):
        if type(outputList[i]) is not list:
           outputList[i] = [outputList[i]] 
    return outputList


def extract_keys(the_dict, prefix=''):
    # TODO
    # fix bug with list of dicts not being extracted
    # but only first element

    key_list = []
    
    for key, value in the_dict.items():
        
        # set the prefix
        if len(prefix) == 0:
            new_prefix = key
        else:
            new_prefix = prefix + '.' + key

        # recursively call extract_keys for nested dicts 
        if type(value) == dict:
            key_list.extend(extract_keys(value, new_prefix))
        elif type(value) == list and type(value[0]) == dict:
            key_list.extend(extract_keys(value[0], new_prefix))
        else:
            key_list.append(new_prefix)


    return key_list

def isHomogeneous(input_dict):

    counter = {}

    for key in input_dict:
        if key not in counter:
            counter[key] = 1
        else:
            counter[key] += 1

    values = list(counter.values())

    print(values)

    # calculate variance

    mean = 0
    for value in values:
        mean += value

    mean /= len(values)

    variance = 0 
    for value in values:
        squared_diff = pow(abs(value - mean), 2)
        variance += squared_diff
    variance /= len(values)

    print("variance:", variance)
    if variance > 3:
        return False
    else:
        return True
        


###

if READ_FILE_FLAG:
    # Read the data in from a text file

    props = []
    datas = []

    # Handling json file input to load as the json object
    with open(sys.argv[2]) as f:
        json_object = json.load(f)

    # Extract the property names from the json object
    props = []
    for i, prop in enumerate(json_object[0]):
        if i > 0:
            props.append(prop)
            datas.append([])

    # Extract data

    for i in range(len(json_object)): 
        for j, prop in enumerate(props):
            datas[j].append(json_object[i][prop])

else:

    # Or read the question and property from the command line and parse the returned data frame
    command = "result = bfq." + sys.argv[2] + ".answer().frame()"
    exec(command)
    # print(result)

    props = sys.argv[4].split('|')
    for i in range(len(props)):
        props[i] = props[i].strip()

    datas = []
    for prop in props:
        data = listify(result[prop])


        overall = {}

        # handle dicts in ACLs 
        for i in range(len(data)):

            item = data[i][0]
            print(type(item))

            if type(item) != str:
                
                if type(data[i][0]) == dict and STEP_TWO_FLAG:

                    exclude_list = sys.argv[6].split(',')
                    for n in range(len(exclude_list)):
                        exclude_list[n] = exclude_list[n].strip()
                        data[i][0].pop(exclude_list[n], None)


                    # # make signature by removing value
                    # data[i][0].pop('name', None)
                    # data[i][0].pop('sourceName', None)
                    # data[i][0].pop('sourceType', None)

                # 
                if type(item) == dict:

                    result = extract_keys(item)
                    # print(result)

                    for element in result:

                        value = item
                        for key in element.split('.'):

                            new_value = value[key]
                            if type(new_value) == list:
                                new_value = new_value[0]

                            value = new_value

                        # print(element, value)
                        if element not in overall: 
                            overall[element] = [value]
                        else:
                            overall[element].append(value)



                data[i][0] = str(data[i][0])

                # print(data[i])


        datas.append(data)


#TEMP
for key, value in overall.items():
    if isHomogeneous(value):
        print(Fore.GREEN + key, ": ", value)
    else:
        print(Fore.RED + key, ": ", value)
    print(Style.RESET_ALL)
    print()
print()

# for d in data:
#     if type(d[0]) == dict:
#         for key,value in d[0].items():
#             print("%s: %s" % (key, value))
#     else:
#         print(d[0])
#     print()


# Encode using multi label binarizer and calculate frequency
mlb = MultiLabelBinarizer()

encodedLists = []
frequencyLists = []
uniqueClasses = []
proportion = 0 


# The for loop encodes each column (aka feature) in a binary format where
# 0 means it a class is absent and 1 means the class is present in that row.
for i, data in enumerate(datas):
    # fit_transform calculates the size of each category automatically based on the input data
    # and then encodes it into the multilabel bit encoding
    encodedList = mlb.fit_transform(datas[i])
    encodedLists.append(encodedList)
    uniqueClasses.append(mlb.classes_)

    frequencyList = [0] * len(encodedList[0])

    proportion += len(encodedList[0]) * len(encodedList)

    for e in encodedList:
        for i in range(len(e)):
            frequencyList[i] += e[i] 
            
    frequencyLists.append(frequencyList)

# Calculate density
# For each matrix, first sum up each column. Then for each row, add the corresponding
# value to the density value if the matching row value is 1.
# Do this for each matrix, and then sum up all those values to get the final density value for the column/feature.

densityLists = []
normalizedDensityLists = []
aggregatedDensityList = [0] * len(encodedLists[0])

for i in range(len(encodedLists)):
    densityList = [0] * len(encodedLists[i])
    normalizedDensityList = [0] * len(encodedLists[i])

    for j in range(len(densityList)):
        for k in range(len(encodedLists[i][j])): 

            densityList[j] += encodedLists[i][j][k] * frequencyLists[i][k]
            normalizedDensityList[j] += encodedLists[i][j][k] * frequencyLists[i][k] / float(proportion)
            aggregatedDensityList[j] += encodedLists[i][j][k] * frequencyLists[i][k]

    densityLists.append(densityList)
    normalizedDensityLists.append(normalizedDensityList)

for i, prop in enumerate(props):
    print("%s: %s" % (prop, datas[i]))
    print()
    print("Unique classes: %s" % uniqueClasses[i])
    print()
    print(encodedLists[i])
    print()

# print(densityLists)
# print(aggregatedDensityList)

'''
Approach 1: Simple Threshold-based outlier detection with outlier threshold 1/3
'''

# Aggregate all the input data
aggregated = []

for i in range(len(datas[0])):
    value = []
    for data in datas:
        for element in data[i]:
            value.append(element)
    
    
    aggregated.append(tuple(value))

# Unique instance counter of the elements.
valueCounterOutCome = Counter(aggregated)
if DEBUG_PRINT_FLAG:
    print("# Unique Instance Counter:", valueCounterOutCome)

mostCommonElement = valueCounterOutCome.most_common(1)
print("# Most common element from the input data:", mostCommonElement)

mostCommonElementSize = valueCounterOutCome.most_common()[0][1]
if DEBUG_PRINT_FLAG:
    print("# Most common element size:", mostCommonElementSize)

totalSizeOfmultiClassSet = len(aggregated)
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
Approach 2: Alternative outlier detection approaches
'''

# Outlier detection libraries

outliers = outlierLibrary.tukey(aggregatedDensityList)
label = 'Tukey\'s method outliers: ' + str(outliers)
print(label)
print('=' * len(label), end='\n\n')
for outlier in outliers:
    print("Outlier index: %d" % outlier)
    for i, data in enumerate(datas):
        print("\t%s: %s" % (props[i], data[outlier]))
    print()
print()

outliers = outlierLibrary.z_score(aggregatedDensityList)
label = 'Z-Score method outliers: ' + str(outliers)
print(label)
print('=' * len(label), end='\n\n')
for outlier in outliers:
    print('Outlier index: %d' % outlier)
    for i, data in enumerate(datas):
        print('\t%s: %s' % (props[i], data[outlier]))
    print()
print()

outliers = outlierLibrary.modified_z_score(aggregatedDensityList)
label = 'Modified Z-Score method outliers: ' + str(outliers)
print(label)
print('=' * len(label), end='\n\n')
for outlier in outliers:
    print('Outlier index: %d' % outlier)
    for i, data in enumerate(datas):
        print('\t%s: %s' % (props[i], data[outlier]))
    print()
print()

cooksDensityList = []
for i, value in enumerate(aggregatedDensityList):
    cooksDensityList.append((i, value))

outliers = outlierLibrary.cooks_distance(cooksDensityList)
label = 'Cook\'s distance method outliers: ' + str(outliers)
print(label)
print('=' * len(label), end='\n\n')
for outlier in outliers:
    print('Outlier index: %d' % outlier)
    for i, data in enumerate(datas):
        print('\t%s: %s' % (props[i], data[outlier]))
    print()
print()

# Calcules the outliers using mahalanobis distance method.
# Then for each outlier, print out the associated information related to
# its features and their values.
outliers = outlierLibrary.mahalanobis_distance(densityLists)
label = 'Malanobis distance method outliers: ' + str(outliers)
print(label)
print('=' * len(label), end='\n\n')
for outlier in outliers:
    print('Outlier index: %d' % outlier)
    for i, data in enumerate(datas):
        print('\t%s: %s' % (props[i], data[outlier]))
    print()
print()


sys.exit(0)


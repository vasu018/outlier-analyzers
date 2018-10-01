import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
import sys
from pybatfish.client.commands import *
from pybatfish.question.question import load_questions, list_questions
from pybatfish.question import bfq
from colorama import Fore, Back, Style

# setup

load_questions()

pd.compat.PY3 = True
PD_DEFAULT_COLWIDTH = 250
pd.set_option('max_colwidth', PD_DEFAULT_COLWIDTH)

bf_init_snapshot('datasets/networks/example')

# Help flag
if len(sys.argv) < 3 or sys.argv[1] == '-h':
    print("#################################################################################")
    print(Fore.RED + "# Error: Invalid arguments !!!")
    print(Style.RESET_ALL)
    print(Fore.BLUE + "# Usage: python3 outlierAnalyzer.py <propertiesType> <propertyAttributes>")
    print("# Examples: python3 outlierAnalyzer.py \"nodeProperties()\" NTP_Servers")
    print("#           python3 outlierAnalyzer.py \"interfaceProperties()\" \"HSRP_GROUPS | MTU\"")
    print(Style.RESET_ALL)
    print("#################################################################################")
    sys.exit(0)

# utility functions

def listify(frame):
    outputList = list(frame)
    for i in range(len(outputList)):
        if type(outputList[i]) is not list:
           outputList[i] = [outputList[i]] 
    return outputList

# Read the question and property from the command line and parse the returned data frame
command = "result = bfq." + sys.argv[1] + ".answer().frame()"
exec(command)
print(result)

props = sys.argv[2].split('|')
for i in range(len(props)):
    props[i] = props[i].strip()

datas = []
for prop in props:
    data = listify(result[prop])
    datas.append(data)


# TODO
# include other information along with the density value, maybe use object-oriented

# Encode using multi label binarizer and calculate frequency
mlb = MultiLabelBinarizer()

encodedLists = []
frequencyLists = []
proportion = 0 



for i, data in enumerate(datas):
    encodedList = mlb.fit_transform(datas[i])
    encodedLists.append(encodedList)

    frequencyList = [0] * len(encodedList[0])

    proportion += len(encodedList[0]) * len(encodedList)

    for e in encodedList:
        for i in range(len(e)):
            frequencyList[i] += e[i] 
            
    frequencyLists.append(frequencyList)

# Calculate density

densityList = [0] * len(encodedLists[0])
normalizedDensityList = [0] * len(encodedLists[0])


for i in range(len(encodedLists)):
    for j in range(len(densityList)):
        for k in range(len(encodedLists[i][j])): 

            densityList[j] += encodedLists[i][j][k] * frequencyLists[i][k]
            normalizedDensityList[j] += encodedLists[i][j][k] * frequencyLists[i][k] / float(proportion)
            

# Outlier detection libraries

import newOutlierLibrary

print(densityList)
print(datas)

outliers = newOutlierLibrary.tukey(densityList)
label = 'Tukey\'s method outliers: ' + str(outliers)
print(label)
print('=' * len(label), end='\n\n')
for outlier in outliers:
    print("Outlier index: %d" % outlier)
    for i, data in enumerate(datas):
        print("\t%s: %s" % (props[i], data[outlier]))
    print()
print()

outliers = newOutlierLibrary.z_score(densityList)
label = 'Z-Score method outliers: ' + str(outliers)
print(label)
print('=' * len(label), end='\n\n')
for outlier in outliers:
    print('Outlier index: %d' % outlier)
    for i, data in enumerate(datas):
        print('\t%s: %s' % (props[i], data[outlier]))
    print()
print()

outliers = newOutlierLibrary.modified_z_score(densityList)
label = 'Modified Z-Score method outliers: ' + str(outliers)
print(label)
print('=' * len(label), end='\n\n')
for outlier in outliers:
    print('Outlier index: %d' % outlier)
    for i, data in enumerate(datas):
        print('\t%s: %s' % (props[i], data[outlier]))
    print()
print()

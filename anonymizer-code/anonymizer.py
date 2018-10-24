#!/usr/bin/env python3
import fileinput
import re
import pandas as pd
import statistics
import numpy
from collections import Counter
import sys
from colorama import Fore, Back, Style

# Help flag
if len(sys.argv) < 2 or sys.argv[1] == '-h':
    print(Fore.BLUE + "########################################################")
    print("# Usage: python3 anonymizer.py <fileInput>")
    print("########################################################")
    print(Style.RESET_ALL)
    sys.exit(0)

try:
    inputFileTxt = sys.argv[1]
    print("Input file name is:", sys.argv[1])
except FileNotFoundError:
    print('Invalid file specified!')
    sys.exit(0)


# Takes the input file argument from the user and anonymizes the name and id parameters 
# in the input file and copies the ouput to the outputAnonymizedFile.txt file.
# This file will be stored in the current directory.

outputAnonymizedFile = "./outputAnonymizedFile.txt"

counter = 0
with open(inputFileTxt) as inputFile:
    with open(outputAnonymizedFile, "w") as outputFileDesc:
        for line in inputFile:
        # In this if condition, the code anonymizes the name and id present in the configuration file 
        # and replace it with nodeid- and nodename- followed by sequential counter value. 
        # [TODO]: We can generate random hash which can be extracted same for the file name. 
            if "\'id\':" and "\'name\':" in line:
                counter = counter+1
                nodeID = "nodeid-" + str(counter)
                nodeName = "nodename-" + str(counter)
                m = re.search('\'id\':\s+\'(.*)\'\,\s+\'name\':\s+\'(.*)\'', line)
                line = line.replace(m.group(1), nodeID)
                line = line.replace(m.group(2), nodeName)
                outputFileDesc.write(line) 
            else:
                outputFileDesc.write(line) 

print("Anonymized output file is stored in:", outputAnonymizedFile)

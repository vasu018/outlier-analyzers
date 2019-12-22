# This script finds the feature dependecy score of different named structures.
# Run the script as follows: python feature_dependency.py <network-json-files-directory>
# The argument contains all the JSON files for the particular network.
# We find occurences of same name in different named structure files. Based on that, we calculate a score for
# every named structure using PageRank algorithm.

import pandas as pd
import numpy as np
from collections import defaultdict
import json
import numpy as np
import os,sys
import networkx as nx

# Get the directory given as command line argument.
directory = sys.argv[1]

# Get all files in the directory
files = os.listdir(directory)

# Dictionary to save the names of named structures.
names_directory = defaultdict(list)
for file in files:
    try:
        df = pd.read_json(os.path.join(directory, file))
    except:
        continue
    # Get the name of the named structure.
    named_structure = file.split(".")[0]
    # Save the names in the given named structure file. In our case, names are the indices of the dataframe.
    names_directory[named_structure] = list(df.index)

# dictionary to save all relations between different named structures.
final_dict = defaultdict(int)

# Iterate over all named structures.
for n1 in names_directory.keys():
    for n2 in names_directory.keys():
        if n1!=n2:
            # Iterate over all names in the named structure.
            for names in names_directory[n1]:
                if names in names_directory[n2]:
                    # Building the relation and it's reverse.
                    temp = str(n1) + " -> " + str(n2)
                    temp2 = str(n2) + " -> "+str(n1)
                    # Check if the reverse relation already exists in the dictionary.
                    if final_dict[temp2]==0:
                        # Increment the counter for number of occurences.
                        final_dict[temp]+=1
                        # This is needed as defaultdict creates an entry even when we check if it exists.
                        del final_dict[temp2]

# Building the graph.
D = nx.Graph()
for k,v in final_dict.items():
    # Getting the two named structures
    [i,j] = k.split("->")
    i=i.strip()
    j=j.strip()
    # Adding a weighted edge.
    D.add_weighted_edges_from([(i,j,v)])
# Getting the pagerank scores of the named structures.
weights = nx.pagerank(D)
print("Feature dependency score: ",weights)
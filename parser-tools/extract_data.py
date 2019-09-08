
#########################################################
#run this script using the following command:
#python extract_data.py <path of netowork config directory> <name of network>
#example:
#python extract_data.py networks/test_network test_network
#Here, networks/test_network contains folder of config files.

#Upon completion, the output should be:

#Creating directory  test_network-json (only if the directory doesn't already exists)
#JSON files saved

#########################################################
import logging
import random
import os
import collections
import pandas as pd
import sys
import networkx as nx
import json
import re

from IPython.display import display
from pandas.io.formats.style import Styler

from pybatfish.client.commands import *
# noinspection PyUnresolvedReferences
from pybatfish.datamodel import Interface, Edge
from pybatfish.datamodel.flow import HeaderConstraints, PathConstraints
from pybatfish.question import bfq, load_questions  # noqa: F401
from pybatfish.util import get_html

load_questions()

NETWORK_NAME = sys.argv[2]#"campus-anon-net1"  #Name of the network
SNAPSHOT_NAME = "example_snapshot"

SNAPSHOT_PATH = sys.argv[1]#"networks/campus-anon-net1" # Path of the config files.

# Now create the network and initialize the snapshot
bf_set_network(NETWORK_NAME)
bf_init_snapshot(SNAPSHOT_PATH, name=SNAPSHOT_NAME, overwrite=True)
load_questions()

data = bfq.namedStructures().answer().frame()

Structure_types = list(data.Structure_Type.unique())

for struct in Structure_types:

    df = data[data['Structure_Type']==struct]

    col_names = list(df.Structure_Name.unique())
    unique_nodes = list(df.Node.unique())

    struct_df = pd.DataFrame(index=unique_nodes,columns=col_names)

    nodes = df['Node']
    acls = df['Structure_Name']
    values = df['Structure_Definition']

    zip_data = zip(nodes,acls,values)

    for index,column,value in zip_data:
        struct_df.loc[index,column] = [value]

    fileName=str(struct)+".json"
    directory = "./"+str(NETWORK_NAME)+"-json"
    if not os.path.exists(directory):
        print("Creating directory ",directory[2:])
        os.mkdir(directory)
    fullName = os.path.join(directory, fileName)
    struct_df.to_json(fullName,orient="index")


nodeProp = bfq.nodeProperties().answer().frame()
directory = "./"+str(NETWORK_NAME)+"-json"
nodeProp.to_json(os.path.join(directory,"nodeProperties.json"),orient = 'records', lines = True)

temp = bfq.edges().answer().frame()
del temp['IPs']
del temp["Remote_IPs"]

def convert(x):
    s = str(x)
    a = s.split(':')
    a[0] = re.sub("\[(.*?)\]","",a[0])
    return a[0]

temp = temp.applymap(convert)

G=nx.from_pandas_edgelist(temp, 'Interface','Remote_Interface')
pr = nx.pagerank(G,alpha = 0.9)
with open(os.path.join(directory,"node_ranks.json"), 'w') as fp:
    json.dump(pr, fp)

print("JSON files saved")

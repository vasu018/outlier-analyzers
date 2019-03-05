#########################################################
#run this script using the following command:
#python ranking_nodes_pagerank.py <path of netowork config directory> <name of network>
#example:
#python ranking_nodes_pagerank.py networks/test_network test_network 
#Here, networks/test_network contains folder of config files.

#Upon completion, the output should be:

# Results stored in test_network.txt

#########################################################


import logging
import random
import os
import collections
import pandas as pd
import sys
import networkx as nx
from IPython.display import display
from pandas.io.formats.style import Styler

from pybatfish.client.commands import *
# noinspection PyUnresolvedReferences
from pybatfish.datamodel import Interface, Edge
from pybatfish.datamodel.flow import HeaderConstraints, PathConstraints
from pybatfish.question import bfq, load_questions  # noqa: F401
from pybatfish.util import get_html

bf_logger.setLevel(logging.WARN)

load_questions()





NETWORK_NAME = sys.argv[2]#"campus-anon-net1"  #Name of the network
SNAPSHOT_NAME = "example_snapshot"

SNAPSHOT_PATH = sys.argv[1]#"networks/campus-anon-net1" # Path of the config files.

# Now create the network and initialize the snapshot
bf_set_network(NETWORK_NAME)
bf_init_snapshot(SNAPSHOT_PATH, name=SNAPSHOT_NAME, overwrite=True)

temp = bfq.edges().answer().frame()
del temp['IPs']
del temp["Remote_IPs"]

def convert(x):
    s = str(x)
    a = s.split(':')
    return a[0]

temp = temp.applymap(convert)
G=nx.from_pandas_edgelist(temp, 'Interface','Remote_Interface')
pr = nx.pagerank(G,alpha = 0.9)
pr = list(pr.items())
pr.sort(key = lambda x:x[1],reverse = True)

with open(NETWORK_NAME+'.txt', 'w') as f:
    for item in pr:
        f.write("%s\n" % str(item))
print('Results stored in ',NETWORK_NAME,'.txt')















#sdsddsdsd

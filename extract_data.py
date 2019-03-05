import logging
import random
import os
import collections
import pandas as pd
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

NETWORK_NAME = "campus-anon-net1"  #Name of the network
SNAPSHOT_NAME = "example_snapshot"

SNAPSHOT_PATH = "networks/campus-anon-net1" # Path of the config files.

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
    directory = "./"+str(NETWORK_NAME)+" json files"
    if not os.path.exists(directory):
        print("creating directory")
        os.mkdir(directory)
    fullName = os.path.join(directory, fileName)
    struct_df.to_json(fullName,orient="index") 

print("JSON files saved")

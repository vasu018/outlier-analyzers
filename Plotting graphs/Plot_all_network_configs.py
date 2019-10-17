import logging
import random
import os
import collections
import pandas as pd
from IPython.display import display
from pandas.io.formats.style import Styler
import networkx as nx
import re
import matplotlib.pyplot as plt
import seaborn as sns
from time import time
from collections import defaultdict
from pybatfish.client.commands import *
from pybatfish.datamodel import Interface, Edge
from pybatfish.datamodel.flow import HeaderConstraints, PathConstraints
from pybatfish.question import bfq, load_questions  # noqa: F401
from pybatfish.util import get_html
import json

load_questions()

pd.set_option('display.max_colwidth', -1)
pd.set_option('display.max_columns', None)
pd.set_option('display.html.use_mathjax', False)

_STYLE_UUID = "pybfstyle"


class MyStyler(Styler):
    """A custom styler for displaying DataFrames in HTML"""

    def __repr__(self):
        return repr(self.data)


def show(df):
    """
    Displays a dataframe as HTML table.

    Replaces newlines and double-spaces in the input with HTML markup, and
    left-aligns the text.
    """

    # workaround for Pandas bug in Python 2.7 for empty frames
    if not isinstance(df, pd.DataFrame) or df.size == 0:
        display(df)
        return
    df = df.replace('\n', '<br>', regex=True).replace('  ', '&nbsp;&nbsp;',
                                                      regex=True)
    display(MyStyler(df).set_uuid(_STYLE_UUID).format(get_html)
            .set_properties(**{'text-align': 'left', 'vertical-align': 'top'}))

plt.rcParams['figure.figsize'] = 15,10
plt.rcParams['xtick.labelsize']=18
plt.rcParams['ytick.labelsize']=18
plt.rcParams["axes.labelsize"] = 20
plt.rcParams["axes.titlesize"] = 20
plt.rcParams["legend.fontsize"] = 10

#Plotting the number of network properties for each network
directories = [x[1] for x in os.walk(os.getcwd())][0]
named_structures = set()
main_d = defaultdict(dict)
for directory in directories:
    curr_d = defaultdict(int)
    NETWORK_NAME = directory
    SNAPSHOT_NAME = "example_snapshot"

    SNAPSHOT_PATH = directory

    # Now create the network and initialize the snapshot
    try:
        bf_set_network(NETWORK_NAME)
        bf_init_snapshot(SNAPSHOT_PATH, name=SNAPSHOT_NAME, overwrite=True)
    except:
        print(directory + " not loaded")
        continue
    load_questions()
    data = bfq.namedStructures().answer().frame()
    Structure_types = list(data.Structure_Type.unique())
    named_structures.update(Structure_types)
    for struct in Structure_types:
    
        df = data[data['Structure_Type']==struct]
        curr_d[struct] = df.size
    main_d[directory] = dict(curr_d)


for network in main_d:
    c = main_d[network]
    for struct in named_structures:
        if struct not in c:
            c[struct] = 0
            
df = pd.DataFrame(main_d).T
df['Total'] = df.sum(axis = 1)
df = df.drop(df[df.Total < 1].index)
df1 = df[df["Total"]>5000]
df2 = df[(df["Total"]<5000) & (df['Total']>1000)]
df3 = df[df["Total"]<1000]
del df1['Total']
del df2['Total']
del df3['Total']

df.plot(kind = "bar", stacked = True,figsize=(15,10))
plt.xlabel("Networks")
plt.yscale("log")
plt.ylabel("Number of properties")
plt.title("Network property distribution (Log Scale)")
plt.savefig("network_property_distribution_log.png")
plt.show()

df1.plot(kind = "bar", stacked = True,figsize=(15,10))
plt.xlabel("Networks")
plt.ylabel("Number of properties")
plt.title("Network property distribution(Large Networks)")
plt.savefig("network_property_distribution0.png")
plt.show()

df2.plot(kind = "bar", stacked = True,figsize=(15,10))
plt.xlabel("Networks")
plt.ylabel("Number of properties")
plt.title("Network property distribution(Medium Sized Networks)")
plt.savefig("network_property_distribution1.png")
plt.show()

df3.plot(kind = "bar", stacked = True,figsize=(15,10))
plt.xlabel("Networks")
plt.ylabel("Number of properties")
plt.title("Network property distribution(Small Networks)")
plt.savefig("network_property_distribution2.png")
plt.show()


#Plotting the number of nodes in each network
directories = [x[1] for x in os.walk(os.getcwd())][0]
num_nodes = defaultdict(int)
for directory in directories:
    NETWORK_NAME = directory
    SNAPSHOT_NAME = "example_snapshot"

    SNAPSHOT_PATH = directory

    # Now create the network and initialize the snapshot
    try:
        bf_set_network(NETWORK_NAME)
        bf_init_snapshot(SNAPSHOT_PATH, name=SNAPSHOT_NAME, overwrite=True)
    except:
        print(directory + " not loaded")
        continue
    load_questions()
    data = bfq.namedStructures().answer().frame()
    num_nodes[directory] = len(data.Node.unique())

df = pd.DataFrame(num_nodes.items())
df.columns = ["Network","Nodes"]

df = df.drop(df[df.Nodes < 1].index)
df1 = df[(df['Nodes']>80) & (df['Nodes']<300)]
df2 = df[df['Nodes']<80]
df11 = df[df['Nodes']>80]

ax = sns.barplot(x="Network", y="Nodes", data=df1)
ax.set_ylabel("Number of nodes")
ax.set_xticklabels(ax.get_xticklabels(),rotation=90)
plt.title("Large Networks")
plt.savefig("Num_of_nodes_large.png")

ax = sns.barplot(x="Network", y="Nodes", data=df2)
ax.set_ylabel("Number of nodes")
ax.set_xticklabels(ax.get_xticklabels(),rotation=90)
plt.title("Small Networks")
plt.savefig("Num_of_nodes_small.png")

directories = [x[1] for x in os.walk(os.getcwd())][0]
time_dict = defaultdict(float)
for directory in directories:
    start = time()
    NETWORK_NAME = directory
    SNAPSHOT_NAME = "example_snapshot"

    SNAPSHOT_PATH = directory

    # Now create the network and initialize the snapshot
    try:
        bf_set_network(NETWORK_NAME)
        bf_init_snapshot(SNAPSHOT_PATH, name=SNAPSHOT_NAME, overwrite=True)
    except:
        print(directory + " not loaded")
        continue
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
    end = time()
    time_dict[directory] = end - start

df = pd.DataFrame(time_dict.items())
times = []
for k,v in num_nodes.items():
    times.append(v)
df["Time required"] = times
df = df.groupby("Time required").agg('mean')
df = df.reset_index()
df.columns = ["Nodes","Time required"]

ax = sns.barplot(x="Nodes", y="Time required", data=df)
plt.ylabel("Time Required(seconds)")
plt.xlabel("number of Nodes")
ax.set_xticklabels(ax.get_xticklabels(),rotation=90)
plt.title("Performance Scalabilty")
plt.savefig("performance_scalability.png")
extract_data.py

-This script will be take config files as input( command line) and create all the necessary json files required for further processing.
It also stores the PageRank rank scores for all the nodes in json file node_rank.json


run this script using the following command:
python extract_data.py <path of netowork config directory> <name of network>
example:
python extract_data.py networks/test_network test_network
Here, networks/test_network contains folder of config files.

Upon completion, the output should be:

Creating directory  test_network json files(only if the directory doesn't already exists)
JSON files saved

Requirements: Batfish and pybatfish(to be installed from the official github page), pandas, networkx

The code will store all the created json files in a directory. Detailed info about the working are in the file.
Pay close attention on how to pass the command line inputs.

==============================================================================================

ranking.py

-This script will take the list of outliers (in json format) and perform the necessary operations related to ranking and severity of the outliers.

running the script:

python ranking.py <node_ranks.py(file with PageRank scores of all the nodes)> <list of all outliers json files>

example:
python ranking.py node_ranks.json signature-ip-access.json signature-route-filter.json signature-route-policy.json

This file currently works on the old format where the entire ACL list was considered as an outlier.
It begins by calculating the metric1,metric2 and metric3 for all the nodes. (Metric_3 is calculated using the statistical approach)

Then it calculates severity for each outlier based on these scores.
There are also functions defined for incorporating user feedback and updating the outlier's severity.

Detailed explanation in the file.

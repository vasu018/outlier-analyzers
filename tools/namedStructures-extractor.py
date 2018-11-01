"""
Extract the named structures from the configuration data frame outcomes from pybatfish.
"""
import sys
from pybatfish.client.commands import *
from pybatfish.question.question import load_questions, list_questions
from pybatfish.question import bfq
import pandas as pd

GRANULARITY_OF_PROPERTY = True 
load_questions()

JSON_OUTPUT = True 

ipaccesslist_file = "./namedStructureProperties_ip-accesslist.json"
routingpolicies_file = "./namedStructureProperties_routepolicies.json"

# Configuration snapshot
bf_init_snapshot('../Mixed_V_CampusNet_Anonymized') 

if len(sys.argv) > 1:
    questionDir = sys.argv[1]
else:
    questionDir = 'questions/experimental'
load_questions(questionDir)

bf_session.printAnswers = True

print("loading the Mixed_V_CampusNet_Anonymized testrig")

def listify(frame):
    outputList = list(frame)
    for i in range(len(outputList)):
        if type(outputList[i]) is not list:
           outputList[i] = [outputList[i]]
    return outputList


named_structures_properties2 = [
    "as-path-access-lists",
    "authentication-key-chains",
    "community-lists",
    "ike-policies",
    "ip-access-lists",
    "ip6-access-lists",
    "ipsec-policies",
    "ipsec-proposals",
    "ipsec-vpns",
    "route-filter-lists",
    "route6-filter-lists",
    "routing-policies",
    "vrfs",
    "zones"]

named_structures_properties = [
    #"ip-access-lists"]
    "routing-policies"]


for i in range(len(named_structures_properties)):
    named_structures_properties[i] = named_structures_properties[i].strip()


datas = []
for named_struct_property in named_structures_properties:
    named_structures_property_frames=  bfq.namedStructures(nodes='.*',properties=named_struct_property).answer().frame()
    named_structure_property_columns = named_structures_property_frames.columns
    print("# Data for Structure Property:", named_struct_property)
    for column in named_structure_property_columns:
        print("# Column data of the property:", column)
        if (JSON_OUTPUT):
            #json_output = named_structures_property_frames[column].to_json(orient='records')[1:-1].replace('},{', '} {')
            json_output = named_structures_property_frames[column].to_json(orient='records')[1:-1]
            print(json_output)
            datas.append(json_output)
        else:
            data = listify(named_structures_property_frames[column])
            print(data)
            #print(named_structures_property_frames[column])
            datas.append(data)

#with open(ipaccesslist_file, 'w') as namedStructFile:
with open(routingpolicies_file, 'w') as namedStructFile:
    index = 0
    for item in datas:
        if index == 0:
            columnName = "ColumnName:" + named_structure_property_columns[index] + "=>"
        else:
            columnName = ";\nColumnName:" + named_structure_property_columns[index] + "=>"
        namedStructFile.write(columnName)
        namedStructFile.write(item)
        index = index +1


"""
Extract the named structures from the configuration data frame outcomes from pybatfish.
"""
import sys
from pybatfish.client.commands import *
from pybatfish.question.question import load_questions, list_questions
from pybatfish.question import bfq
import pandas as pd

DEBUG_PROPERTY = True 
load_questions()

# Configuration snapshot
bf_init_snapshot('../../pybatfish-new-clone/Viszisas_Anonymized') 

if len(sys.argv) > 1:
    questionDir = sys.argv[1]
else:
    questionDir = 'questions/experimental'
load_questions(questionDir)

bf_session.printAnswers = True

print("loading the Viszisas_Anonymized testrig")

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
    "ip-access-lists"]


for i in range(len(named_structures_properties)):
    named_structures_properties[i] = named_structures_properties[i].strip()


# Extracting the data specific to each named Structure property.
datas = []
#with open('./namedStructureProperties.json', 'w') as namedStructFile:
if (DEBUG_PROPERTY):
    for named_struct_property in named_structures_properties:
        print("Named Structure Property is:", named_struct_property)
        named_structures_property_frame =  bfq.namedStructures(nodes='.*',properties=named_struct_property).answer().frame()
        named_structure_property_columns = named_structures_property_frame.columns
        for column in named_structure_property_columns:
            print(named_structures_property_frame)
            datas.append(named_structures_property_frame)

# Extract the data specific at granularity of each instance of the property.
else:
    named_structures_properties_all =  bfq.namedStructures(nodes='.*').answer().frame()
    for prop in named_structures_properties_all.columns:
        print("Property is:", prop)
        data = listify(named_structures_properties_all[prop])
        print(named_structures_properties_all[prop])
        #print(data)
        datas.append(data)
        namedStructFile.write("%s\n" % data)

#close(namedStructFile)
#print(datas)
#with open('./namedStructureProperties.json', 'w') as namedStructFile:
#    for item in datas:
#        namedStructFile.write("%s\n" % item)


#named_structures_frame =  bfq.namedStructures(nodes='.*',properties="routing-policies").answer().frame()
#
#print("Frame for routing-policies:")
#print(named_structures_frame)

#datas = []
#for prop in props:
#    data = listify(named_structures_frame[prop])
#    datas.append(data)



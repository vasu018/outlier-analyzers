import sys
import re
import copy
import math
import json
import ast
import random
import numpy as np
import pandas as pd
from collections import abc
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MultiLabelBinarizer
from scipy.spatial.distance import cdist, pdist
from colorama import Fore, Back, Style
import copy
import time

# Error Message
def error_msg():
    print("****************************")
    print(Fore.BLUE, end='')
    print("Invalid flags sent!")
    print(Style.RESET_ALL, end='')
    print("****************************")
    sys.exit(0)

edit_key = ""
edit_value = ""
ACTION_FLAG = 0
# Reading Data
try:
    flag_file = '.flag_'+sys.argv[2]
    if sys.argv[1] == "-j":
        df = pd.read_json(sys.argv[2],orient = "index")
        try:
            if sys.argv[3] == "-a":
                ACTION_FLAG = 3
        except:
            ACTION_FLAG = 0
        f = open(flag_file,'w')
        f.write('{}'.format(ACTION_FLAG))
        f.close()
    elif sys.argv[1] == "-e":
        df = pd.read_json(sys.argv[2],orient = "index")
        edit_key = sys.argv[3]
        edit_value = sys.argv[4]
        ACTION_FLAG = 1
    elif sys.argv[1] == "-d":
        df = pd.read_json(sys.argv[2],orient = "index")
        ACTION_FLAG = 2
    else:
        error_msg()
except:
    error_msg()

outlier_filename = 'outlier_'+sys.argv[2]
cluster_filename = '.cluster_'+sys.argv[2]
sig_filename = '.sig_'+sys.argv[2]
outlier_nodes_filename = '.outlier_nodes_'+sys.argv[2]
print("===========================================================")
print(Fore.BLUE, end='')
print("outlier-analyzer code started ...")
print(Style.RESET_ALL)
print(Fore.GREEN, end='')

# *****************************************************************************
# *****************************************************************************
# Helper Methods Start

# Perform K-Means Clustering
def perform_kmeans_clustering(df):
    labels = []
    features = df[df.columns]
    # TODO: Check for k-range
    k_select = 13
    kmeans = KMeans(n_clusters = k_select)
    kmeans.fit(features)
    kmeans_centroids = kmeans.cluster_centers_
    labels = kmeans.labels_
    df["kmeans_cluster_number"] = pd.Series(labels)

def extract_keys(the_dict, prefix=''):
    key_list = []

    for key, value in the_dict.items():

        if len(prefix) == 0:
            new_prefix = key
        else:
            new_prefix = prefix + '=' + key
            
        try:
            if type(value) == dict:
                key_list.extend(extract_keys(value, new_prefix))
            elif type(value) == list and type(value[0]) == dict:
                key_list.extend(extract_keys(value[0], new_prefix))
            elif type(value) == list and type(value[0]) != dict:
                key_list.append(new_prefix)
            else:
                key_list.append(new_prefix)
        except:
            key_list.append(new_prefix)

    return key_list

def calculate_z_score(arr):
    
    z_score = []
    
    mean = np.mean(arr)
    std = np.std(arr)
    if(std == 0):
        return np.ones(len(arr))*1000
    for val in arr:
        z_score.append((val-mean)/std)
        
    return z_score

# Calculating the overall dictionary
def overall_dict(data_final):
    overall_array = []
    for data in data_final:
        overall = {}
        value = None
        result = None
        new_value = None
        for item in data:
            if item[0] is None:
                continue
            result = extract_keys(item[0])
            for element in result:
                value = item[0]
                for key in element.split("="):
                    new_value = value[key]
                    if type(new_value) == list:
                        if len(new_value) != 0:
                            new_value = new_value[0]
                        else:
                            new_value = "#BUG#"

                    value = new_value
                if element not in overall:
                    overall[element] = {}

                if value not in overall[element]:
                    overall[element][value] = 1
                else:
                    overall[element][value] += 1

        overall_array.append(overall)
        overall = {}
    return overall_array

# Calculating the Signature
def calculate_signature_d(overall_arr, index, data_final):
    signature = {}
    for key, value in overall_arr.items():
        sig_threshold = 1.5
        bug_threshold = -1.5
        key_points = []
        data_points = []
        sig_values = []
        for k, v in value.items():
            key_points.append(k)
            data_points.append(v)

        if(len(data_points) == 1):
            sig_values.append((key_points[0], (data_points[0]/len(data_final[index]))))
        # Check for two data points case
        else:
            z_score = calculate_z_score(data_points)
            max_z_score = max(z_score)
            bug_threshold = bug_threshold + (max_z_score - sig_threshold)
            for i in range(len(z_score)):
                if z_score[i] == 1000.0:
                    sig_values.append((key_points[i], "*", (data_points[i]/len(data_final[index]))))
                elif z_score[i] >= sig_threshold:
                    sig_values.append((key_points[i], (data_points[i]/len(data_final[index]))))
                elif z_score[i] <= bug_threshold:
                    sig_values.append((key_points[i], "!", (data_points[i]/len(data_final[index]))))
                elif (z_score[i] < sig_threshold) and (z_score[i] > bug_threshold):
                    sig_values.append((key_points[i], "*", (data_points[i]/len(data_final[index]))))

        if key in signature:
            signature[key].append(sig_values)
        else:
            signature[key] = []
            signature[key]+=(sig_values)
        
    return signature

# Helper Function for Scoring the Signature
def transform_data(data):
    count = 1
    overall = {}
    flag = 0
    i = 0
    while i < count:
        value = None
        result = None
        new_value = None
        for item in data:
            result = extract_keys(item)
            for element in result:
                value = item
                for key in element.split("="):
                    if key in value:
                        new_value = value[key] 
                    if (type(new_value) == list) and (len(new_value) > 1):
                        if flag == 0:
                            count = len(new_value)
                            flag = 1
                        new_value = new_value[i]
                    elif (type(new_value) == list) and (len(new_value) == 1):
                        new_value = new_value[0]
                    value = new_value
                    
                if element not in overall:
                    overall[element] = {}
                
                if type(value) != dict:
                    if value not in overall[element]:
                        overall[element][value] = 1
        i += 1 
    return overall

# Scoring Signature
def calculate_signature_score(signature):
    score_arr = []
    for sig in signature:
        score = 0
        for key, value in sig.items():
            for val in value:
                if (val[1] != "!") and (val[1] != "*"):
                    score += val[1]
        score_arr.append(score)
    return score_arr

# Calculating ACLs similarity Score
def calculate_acl_scores(data_final, all_signatures):
    deviant_arr = []
    count_arr = []
    acls_dict = {}
    acls_arr = []
    sig_score = []
    dev_score = []
    i = 0
    flag = 0
    for acl_list in data_final:
        signature = all_signatures[i]
        for acl in acl_list:
            flag = 0
            if str(acl[0]) not in acls_dict:
                acls_dict[str(acl[0])] = 1
                acls_arr.append(acl[0])
                flag = 1
            else:
                continue
            sig_score.append(signature_scores[i])
            deviant = []
            count = 0
            dev_c = 0
            data = transform_data(acl)
            for data_key, data_val in data.items():
                if data_key in signature:
                    # Key Valid. Now check for actual Value
                    for val in data_val.items():
                        for sig_val in signature[data_key]:
                            if val[0] == sig_val[0]:
                                # value also present. Now check if value part of bug/sig/skip
                                if sig_val[1] == "!":
                                    dev_c += sig_val[2]
                                    deviant.append((data_key, sig_val[0]))
                                elif sig_val[1] == "*":
                                    continue
                                else:
                                    count += sig_val[1]
                else:
                    # Deviant Key
                    deviant.append(data_key)
            if flag == 1:
                count_arr.append(count)
                deviant_arr.append(deviant)
                dev_score.append(dev_c)
        i += 1
    return deviant_arr, count_arr, dev_score, acls_arr, sig_score

def calculate_similarity_score():
    ratio_arr = []
    bug_arr = []
    for i in range(len(acls_arr)):
        try:
            temp = count_arr[i]/sig_score[i]
            if len(deviant_arr[i]) != 0:
                bug_arr.append(True)
            else:
                bug_arr.append(False)
        except:
            bug_arr.append(True)
            temp = 0
        ratio_arr.append(temp)
    return ratio_arr, bug_arr

def get_final_outlier_nodes():
    outlier_list = set()
    nodes_dict = dict(zip(list(df.index), np.arange(0, len(list(df.index)))))
    for i in range(len(acls_arr)):
        if bug_arr[i] == True:
            for j in range(len(nodes_arr[i])):
                outlier_list.add(nodes_dict[nodes_arr[i][j]])
        else:
            continue
    return list(outlier_list)

# Helper Methods End 
# *****************************************************************************
# *****************************************************************************

start = time.time()
# Calculating outliers selected

f = open(flag_file, 'r')
flag = f.readline()
f.close()

# Parsing Data from Dataframe
for index, row in df.iterrows():
    for col in df.columns:
        if row[col] != None:
            row[col][0]['name'] = row[col][0]['name']+':'+str(index)

acl_list_arr = []
for col in df.columns:
    for i in range(len(df[col])):
        if df[col][i] != None:
            temp = df[col][i][0] 
            acl_list_arr.append(temp)
        else:
            continue

if (ACTION_FLAG != 3) and (flag == '0'):
    print("Inside1")
    acl_dict = {}
    acl_arr = []
    node_name_dict = {}
    for acl_list in acl_list_arr:
        name = acl_list['name'].split(":")[0]
        node = acl_list['name'].split(":")[1]
        del acl_list['name']
        acl = json.dumps(acl_list, sort_keys=True)
        if acl not in acl_dict:
            try:
                flag = 0
                #Checking if actually unique or just jumbled
                for dump_acl in acl_arr:
                    if dump_acl[0] == name:
                        #Check conditions of ACLs
                        temp_acl = json.loads(dump_acl[1])
                        main_acl = json.loads(acl)
                        if(len(temp_acl['lines']) != len(main_acl['lines'])):
                            flag = 0
                            continue
                        total = len(temp_acl['lines'])
                        count = 0
                        for sub_acl in temp_acl['lines']:
                            for sub_acl_main in main_acl['lines']:
                                if json.dumps(sub_acl, sort_keys=True) == json.dumps(sub_acl_main, sort_keys=True):
                                    count += 1
                        if count == total:
                            #ACLs are equal, just jumbled
                            flag = 1
                            break

                if flag == 0:
                    acl_dict[acl] = set()
                    acl_dict[acl].add(name)
                    acl_arr.append((name, acl))
                    node_name_dict[acl] = set()
                    node_name_dict[acl].add(node)
                else:
                    acl_arr.append((name, acl))
            except:
                acl_dict[acl] = set()
                acl_dict[acl].add(name)
                acl_arr.append((name, acl))
                node_name_dict[acl] = set()
                node_name_dict[acl].add(node)
        else:               
            acl_dict[acl].add(name)
            acl_arr.append((name, acl))

elif (ACTION_FLAG != 0) and (flag != '0'):
    print("Inside2")
    acl_dict = {}
    acl_arr = []
    node_name_dict = {}
    flag_set = set()
    for acl_list in acl_list_arr:
        name = acl_list['name'].split(":")[0]
        node = acl_list['name'].split(":")[1]
        del acl_list['name']
        acl = json.dumps(acl_list, sort_keys=True)
        try:
            temp = acl_list['lines']
            
            try:
                # Divide into ACLs
                try:
                    temp = acl_list['lines'][0]['name']
                    flag_set.add(1)
                    for acl in acl_list['lines']:
                        temp_name = name + ":" + acl['name']
                        del acl['name']
                        temp_acl = json.dumps(acl, sort_keys=True)
                        if temp_acl not in acl_dict:
                            node_name_dict[temp_acl] = set()
                            node_name_dict[temp_acl].add(node)
                            
                            acl_dict[temp_acl] = set()
                            acl_dict[temp_acl].add(temp_name)
                        else:
                            node_name_dict[temp_acl].add(node)
                            #del acl['name']
                            acl_dict[temp_acl].add(temp_name)
                except:
                    flag_set.add(2)
                    for acl in acl_list['lines']:
                        temp_acl = json.dumps(acl, sort_keys=True)
                        if temp_acl not in acl_dict:
                            acl_dict[temp_acl] = set()
                            acl_dict[temp_acl].add(name)
                            node_name_dict[temp_acl] = set()
                            node_name_dict[temp_acl].add(node)
                        else:
                            acl_dict[temp_acl].add(name)
                            node_name_dict[temp_acl].add(node)
                
            except:
                # Divide into ACL Lists
                flag_set.add(3)
                if acl not in acl_dict:
                    flag = 0
                    #Checking if actually unique or just jumbled
                    for dump_acl in acl_arr:
                        if dump_acl[0] == name:
                            #Check conditions of ACLs
                            temp_acl = json.loads(dump_acl[1])
                            main_acl = json.loads(acl)
                            if(len(temp_acl['lines']) != len(main_acl['lines'])):
                                flag = 0
                                continue
                            total = len(temp_acl['lines'])
                            count = 0
                            for sub_acl in temp_acl['lines']:
                                for sub_acl_main in main_acl['lines']:
                                    if json.dumps(sub_acl, sort_keys=True) == json.dumps(sub_acl_main, sort_keys=True):
                                        count += 1
                            if count == total:
                                #ACLs are equal, just jumbled
                                flag = 1
                                break

                    if flag == 0:
                        acl_dict[acl] = set()
                        acl_dict[acl].add(name)
                        acl_arr.append((name, acl))
                    else:
                        acl_arr.append((name, acl))
                else:               
                    acl_dict[acl].add(name)
                    acl_arr.append((name, acl))
        except:
            flag_set.add(4)
            # RouteFilterList
            if acl not in acl_dict:
                flag = 0
                #Checking if actually unique or just jumbled
                for dump_acl in acl_arr:
                    if dump_acl[0] == name:
                        #Check conditions of ACLs
                        temp_acl = json.loads(dump_acl[1])
                        main_acl = json.loads(acl)
                        if(len(temp_acl['statements']) != len(main_acl['statements'])):
                            flag = 0
                            continue
                        total = len(temp_acl['statements'])
                        count = 0
                        for sub_acl in temp_acl['statements']:
                            for sub_acl_main in main_acl['statements']:
                                if json.dumps(sub_acl, sort_keys=True) == json.dumps(sub_acl_main, sort_keys=True):
                                    count += 1
                        if count == total:
                            #ACLs are equal, just jumbled
                            flag = 1
                            break

                if flag == 0:
                    acl_dict[acl] = set()
                    acl_dict[acl].add(name)
                    acl_arr.append((name, acl))
                else:
                    acl_arr.append((name, acl))
            else:               
                acl_dict[acl].add(name)
                acl_arr.append((name, acl))
        

acl_name_arr = []
datas = []
for key, val in acl_dict.items():
    acl_name_arr.append(val)
    datas.append(key)

print("data parsing done...")


if (ACTION_FLAG == 0) or (ACTION_FLAG == 3):
    # Encoding Categorical data
    mlb = MultiLabelBinarizer()
    encodedLists = []
    frequencyLists = []
    uniqueClasses = []
    proportion = 0 

    for i, data in enumerate(datas):
        encodedList = mlb.fit_transform(datas[i])
        encodedLists.append(encodedList)
        
    data_df = []
    for i in range(len(encodedLists)):
        data_df.append(encodedLists[i][0])
        

    df_enc = pd.DataFrame(data_df)
    df_enc = df_enc.fillna(1000)
    print("data encoding done...")

    # Perform K-Means
    perform_kmeans_clustering(df_enc)
    print("data clustering done...")

    # Grouping data based on their Clusters
    cluster_range = np.arange(13)
    data_final = []
    data_final_enc = []
    for index in cluster_range:
        temp = []
        temp_enc = []
        for i in range(len(df_enc)):
            if df_enc['kmeans_cluster_number'][i] == index:
                temp.append([json.loads(datas[i])])
                temp_enc.append([datas[i]])
        data_final.append(temp)
        data_final_enc.append(temp_enc)

    # Writing Clustered Data into a file
    with open(cluster_filename, 'w') as f:
        f.write(json.dumps(data_final))
    # print("cluster data written to file...")

    # Calculating Overall Structure per Cluster
    overall_array = overall_dict(data_final)

    # Generating Signatures
    all_signatures = []
    for i in range(len(overall_array)):
        signature = calculate_signature_d(overall_array[i], i, data_final)
        all_signatures.append(signature)
    print("signature creation done...")

# Re-Tuning the Signature
elif ACTION_FLAG == 1:

    all_signatures = []
    try:
        with open(sig_filename, 'r') as f:  
            for item in f:
                all_signatures.append(json.loads(item))
    except FileNotFoundError:
        print(Fore.RED, end='')
        print("\nERROR: Calculate outliers on this data first!\n")
        print(Style.RESET_ALL)
        print("__________________________________")
        print(Fore.RED, end='')
        print("outlier-analyzer code failed #")
        print(Style.RESET_ALL)
        print("__________________________________")
        sys.exit()

    # print("original signature retrieved...")

    flag = 0
    all_signatures = all_signatures[0]
    for signature in all_signatures:
        if edit_key in signature:
            for j in range(len(signature[edit_key])):
                if edit_value in signature[edit_key][j][0]:
                    if signature[edit_key][j][1] == "!" or signature[edit_key][j][1] == "*":
                        try:
                            temp = (edit_value, signature[edit_key][j][2])
                            signature[edit_key][j] = temp
                            flag = 1
                        except Exception as e:
                            print(e)

    if flag == 0:
        print(Fore.RED, end='')
        print("\nERROR : Specified Attributes either\n\tnot present or not a bug!")
        print(Style.RESET_ALL, end='')
        print("__________________________________")
        print(Fore.RED, end='')
        print("outlier-analyzer code failed #")
        print(Style.RESET_ALL, end='')
        print("__________________________________")
        sys.exit(0)

    print("signature re-tuning done...")

    data_final = []
    with open(cluster_filename, 'r') as f:  
        for item in f:
            data_final.append(json.loads(item))
    data_final = data_final[0]

# Displaying the Outlier Nodes
elif ACTION_FLAG == 2:

    outlier_nodes_arr = []
    try:
        with open(outlier_nodes_filename, 'r') as f:  
            for item in f:
                outlier_nodes_arr.append(json.loads(item))
    except FileNotFoundError:
        print(Fore.RED, end='')
        print("\nERROR: Calculate outliers on this data first!\n")
        print(Style.RESET_ALL)
        print("__________________________________")
        print(Fore.RED, end='')
        print("outlier-analyzer code failed #")
        print(Style.RESET_ALL)
        print("__________________________________")
        sys.exit()

    # print("outliers list retrieved...")
    print(Style.RESET_ALL)
    print("########################")
    print("Outlier Nodes are:")
    outlier_nodes_arr = outlier_nodes_arr[0]
    print(Fore.RED, end='')
    print(*outlier_nodes_arr, sep="\n")
    print(Style.RESET_ALL)
    print("########################")
    sys.exit(0)


# Scoring Signature
signature_scores = calculate_signature_score(all_signatures)
print("signature scoring done...")

# Scoring ACLs
deviant_arr, count_arr, dev_score, acls_arr, sig_score = calculate_acl_scores(data_final, all_signatures)
print("acl scoring done...")

# Calculate outlier nodes
outlier_nodes = set()
for i in range(len(deviant_arr)):
    if len(deviant_arr[i]) > 0:
        temp = json.dumps(acls_arr[i], sort_keys=True)
        for item in node_name_dict[temp]:
            outlier_nodes.add(item)

with open(outlier_nodes_filename, 'w') as f:
    f.write(json.dumps(list(outlier_nodes)))

# writing all signature to a hidden file
with open(sig_filename, 'w') as f:
    f.write(json.dumps(all_signatures))  

acl_names = []
for k,v in acl_dict.items():
    temp = acl_dict[k]
    acl_names.append(temp)

# Creating dataframe and exporting as a json file
df_final = pd.DataFrame()
df_final['acl_name'] = acl_names
df_final['acl_structure'] = acls_arr
df_final['deviant_properties'] = deviant_arr
df_final['similarity_score'] = count_arr
df_final['deviant_score'] = dev_score
df_final['max_sig_score'] = sig_score

df_final.to_json(outlier_filename, orient='split', index=False)
print(Style.RESET_ALL, end="")

end = time.time()

print("###")
print(Fore.BLUE, end='')
print("OUTLIER-ANALYZER SUCCESSFUL #")
print("time to run : {} seconds".format(round(end - start), 3))
print(Style.RESET_ALL, end='')
print()

# Final Outliers 
print("###########################################################")
print(outlier_nodes)
print(Fore.BLUE, end='')
print("\nTotal Outliers Count = {}".format(len(outlier_nodes)))
print(Style.RESET_ALL, end='')
print("\nTo view the detailed report, open the")
print("json file named: '{}'\n".format(outlier_filename))
print("###########################################################")

sys.exit(0)

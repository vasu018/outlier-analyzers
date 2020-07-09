import sys
import re
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import MultiLabelBinarizer
from scipy.spatial.distance import cdist
from colorama import Fore, Style
from kneed import KneeLocator
import copy
import time
import pickle
import os


def error_msg(error_msg, arg):
    """
    Helper function to display error message on the screen.
    Input:
        The error message along with its respective argument.
        (Values include - filename, selected action).
    Output:
        The formatted error message on the screen along with the argument.
    """
    print("****************************")
    print(Fore.RED, end='')
    print(error_msg,":", arg)
    print(Style.RESET_ALL, end='')
    print("****************************")
    sys.exit(0)


def printINFO(info):
    """
    Helper function to ask the user for Input.
    Input:
        The message that is to be displayed.
    Output:
        The formatted message on the screen.
    """
    print(Fore.BLUE, end='')
    print(info)
    print(Style.RESET_ALL, end='')

# *****************************************************************************
# *****************************************************************************
# Helper Methods Start


def calculate_num_clusters(df, acl_weights):
    """
    Calculates the optimal number of clusters using the elbow_graph approach.
    Input:
       The Pandas dataframe of the input file (ACL.json)
    output:
       The value of k that provides the least MSE.
    """
    files = ['IP_Access_List', 'Route_Filter_List', 'VRF', 'AS_Path_Access_List',
             'IKE_Phase1_Keys', 'IPsec_Phase2_Proposals', 'Routing_Policy']
    k_select_vals = [41, 17, 42, 5, 3, 2, 58]

    curr_file = file_name.split(".")[0]

    file_index = files.index(curr_file)
    return k_select_vals[file_index]

    features = df[df.columns]
    ran = min(len(df.columns), len(discrete_namedstructure))
    if ran > 50:
        k_range = range(1, 587)
    else:
        k_range = range(1, ran)
    print(k_range)
    k_range = range(1, 580)
    distortions = []
    np.seed = 0
    clusters_list = []
    f = open('distortions.txt', 'w')
    for k in k_range:
        print(k)
        kmeans = KMeans(n_clusters=k).fit(features, None, sample_weight=acl_weights)
        clusters_list.append(kmeans)
        cluster_centers = kmeans.cluster_centers_
        k_distance = cdist(features, cluster_centers, "euclidean")
        distance = np.min(k_distance, axis=1)
        distortion = np.sum(distance)/features.shape[0]
        distortions.append(distortion)
        f.write(str(distortion))
        f.write("\n")

    kn = KneeLocator(list(k_range), distortions, S=3.0, curve='convex', direction='decreasing')
    print("Knee is: ", kn.knee)
    plt.xlabel('k')
    plt.ylabel('Distortion')
    plt.title('The Elbow Method showing the optimal k')
    plt.plot(k_range, distortions, 'bx-')
    plt.vlines(kn.knee, plt.ylim()[0], plt.ylim()[1], linestyles='dashed')
    plt.show()
    if kn.knee is None:
        if ran < 5:
            return ran - 1
        else:
            return 5


    return kn.knee


    '''
    for i in range(1, len(avg_within)):
        if (avg_within[i-1] - avg_within[i]) < 1:
            break
    # return i-1 if len(avg_within) > 1 else 1
    # return i - 1 if i > 1 else 1
    '''


def perform_kmeans_clustering(df, ns_weights):
    """
    To get a mapping of the rows into respective clusters generated using the K-means algorithm.
    Input:
        df:The Pandas data-frame of the input file (ACL.json)
        ns_weights: The weights of each name structure which allows the weighted k-means algorithm to work.
    Output:
        Adding respective K-means cluster label to the input dataframe.
        Example:
        Row1 - Label 0 //Belongs to Cluster 0
        Row2 - Label 0 //Belongs to Cluster 0
        Row3 - Label 1 //Belongs to Cluster 1
    """
    global k_select
    k_select = calculate_num_clusters(df, ns_weights)
    features = df[df.columns]
    kmeans = KMeans(n_clusters=k_select)
    kmeans.fit(features, None, sample_weight=ns_weights)
    labels = kmeans.labels_
    df["kmeans_cluster_number"] = pd.Series(labels)


def extract_keys(the_dict, prefix=''):
    """
    Recursive approach to gather all the keys that have nested keys in the input file.
    Input:
        The dictionary file to find all the keys in.
    Output:
        All the keys found in the nested dictionary.
        Example:
        Consider {key1:value1, key2:{key3:value3}, key4:[value4], key5:[key6:{key7:value7}]}
        The function returns key2, key5=key6
    """
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


def get_uniques(data):
    """
    A helper function to get unique elements in a List.
    Input:
        A list that we need to capture uniques from.
    Output:
        A dictionary with unique entries and count of occurrences.
    """
    acl_count_dict = {}

    for acl in data:
        acl = json.dumps(acl)
        if acl not in acl_count_dict:
            acl_count_dict[acl] = 1
        else:
            value = acl_count_dict[acl]
            value += 1
            acl_count_dict[acl] = value

    keys = []
    values = []
    for key, value in acl_count_dict.items():
        keys.append(key)
        values.append(value)
    return keys, values


def overall_dict(data_final):
    """
    Parses through the dictionary and appends the frequency with which the keys occur.
    Input:
        A nested dictionary.
        Example:
            {key1:{key2:value1, key3:value2, key4:{key5:value3}}
            {key6:{key7:value2}
            {key8:{key3:value3, key4:value5, key6:value3}}
    Output:
        Returns a new array with the nested keys appended along with a tuple containing the un-nested value along with
        the frequency count.
        [{
        key1=key2:{'value1':1},
        key1=key3:{'value2':2},
        key1=key4=key5:{'value3':3},
        key6=key7:{'value2':2},
        key8=key3:{'value3':3},
        key8=key4:{'value5':1},
        key8=key6:{'value3':1}
       }]
    """

    overall_array = []
    for data in data_final:
        overall = {}
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

    return overall_array


def get_overall_dict(data_final):
    """
    Parses through the dictionary and appends the frequency with which the keys occur.
    Input:
        A nested dictionary.
        Example:
            {key1:{key2:value1, key3:value2, key4:{key5:value3}}
            {key6:{key7:value2}
            {key8:{key3:value3, key4:value5, key6:value3}}
    Output:
        Returns a new array with the nested keys appended along with a tuple containing the unnested value along with the frequency count.
        [{
        key1=key2:{'value1':1},
        key1=key3:{'value2':2},
        key1=key4=key5:{'value3':3},
        key6=key7:{'value2':2},
        key8=key3:{'value3':3},
        key8=key4:{'value5':1},
        key8=key6:{'value3':1}
       }]
    """

    overall_array = []
    for data in data_final:
        overall = {}
        new_value = None
        flag = 0
        for item in data:
            visited = {"lines=name":1}
            if item[0] is None:
                continue
            result = extract_keys(item[0])
            for element in result:
                value = item[0]
                for key in element.split("="):
                    if element not in visited:
                        visited[element] = 1
                        new_value = value[key]
                        flag = 0
                        if type(new_value) == list:
                            if len(new_value) > 0:
                                for list_data in new_value:
                                    if element not in overall:
                                        overall[element] = {}
                                    temp = element
                                    temp_val = list_data
                                    temp = temp.split("=", 1)[-1]
                                    while len(temp.split("=")) > 1:
                                        temp_val = temp_val[temp.split("=")[0]]
                                        temp = temp.split("=", 1)[-1]

                                    list_key = temp
                                    check = 0
                                    try:
                                        if type(temp_val[list_key]) == list:
                                            if temp_val[list_key][0] not in overall[element]:
                                                overall[element][temp_val[list_key][0]] = 1
                                                check = 1
                                        else:
                                            if temp_val[list_key] not in overall[element]:
                                                overall[element][temp_val[list_key]] = 1
                                                check = 1
                                    except:
                                        dummy=0
                                        '''
                                        do nothing
                                        '''
                                    try:
                                        if check == 0:
                                            if type(temp_val[list_key]) == list:
                                                if temp_val[list_key][0] in overall[element]:
                                                    overall[element][temp_val[list_key][0]] += 1
                                            else:
                                                if temp_val[list_key] in overall[element]:
                                                    overall[element][temp_val[list_key]] += 1
                                    except:
                                        dummy=0

                                    flag = 1
                                value = new_value

                        else:
                            '''
                            Type is not list
                            '''
                            value = new_value

                    else:
                        if flag == 0:
                            if element not in overall:
                                overall[element] = {}

                            if new_value not in overall[element]:
                                overall[element][new_value] = 1
                            else:
                                overall[element][new_value] += 1

                if flag == 0:
                    if element not in overall:
                        overall[element] = {}

                    if new_value not in overall[element]:
                        overall[element][new_value] = 1
                    else:
                        overall[element][new_value] += 1

        overall_array.append(overall)

    return overall_array


def calculate_z_score(arr):
    """
    Calculates the Z-score (uses mean) (or) Modified Z-score (uses median) of data-points
    Input:
        Data points generated from parsing through the input file.
        Also considers the Z_SCORE_FLAG that is set previously with 0 (default) using the Modified Z-score and 1 using Z-score.
    Output:
        The Z-score of given data-points array.
    """

    if len(arr) == 1:
        return arr

    z_score = []

    '''
        Calculates the Z-score using mean. Generally used if distribution is normal (Bell curve).
    '''

    if Z_SCORE_FLAG:
        mean = np.mean(arr)
        std = np.std(arr)
        if std == 0:
            return np.ones(len(arr)) * 1000
        for val in arr:
            z_score.append((val - mean) / std)

        '''
        Modified Z-score approach.
        Calculates the Z-score using median. Generally used if distribution is skewed.
        '''
    else:
        median_y = np.median(arr)
        medians = [np.abs(y - median_y) for y in arr]
        med = np.median(medians)

        median_absolute_deviation_y = np.median([np.abs(y - median_y) for y in arr])
        if median_absolute_deviation_y == 0:
            return np.ones(len(arr)) * 1000
        z_score = [0.6745 * (y - median_y) / median_absolute_deviation_y for y in arr]

    return z_score


def calculate_signature_d(overall_arr):
    """
    Uses Z-score to generate the signatures of data-points and also maps points on level of significance (include for
    signature calculation, include for bug calculation, no significance).
    If Z-score is equal to 1000.0 or in between sig_threshold and bug_threshold, no-significance.
    If Z-score is >= sig_threshold, include for signature calculation.
    If Z-score is <= bug_threshold, include for bug calculation.
    Input:
        The individual master-signature generated for each Cluster.
    Output:
        An array containing dictionaries marked with tags that represent the action that needs to be performed on them.
    """
    signature = {}
    for key, value in overall_arr.items():
        sig_threshold = 0.5
        bug_threshold = -0.1
        key_points = []
        data_points = []
        sig_values = []

        for k, v in value.items():
            key_points.append(k)
            data_points.append(v)

        if len(data_points) == 1:
            sig_values.append((key_points[0], (data_points[0])))
            '''
            Check for two data points case
            '''
        else:
            z_score = calculate_z_score(data_points)
            if len(z_score) > 0:
                avg_z_score = sum(z_score)/len(z_score)
                bug_threshold = bug_threshold + (avg_z_score - sig_threshold)
            for i in range(len(z_score)):
                present_zscore = z_score[i]
                if present_zscore == 1000.0:
                    sig_values.append((key_points[i], "*", (data_points[i])))
                elif present_zscore >= sig_threshold:
                    sig_values.append((key_points[i], (data_points[i])))
                elif present_zscore <= bug_threshold:
                    sig_values.append((key_points[i], "!", (data_points[i])))
                elif (present_zscore < sig_threshold) and (present_zscore > bug_threshold):
                    sig_values.append((key_points[i], "*", (data_points[i])))

        if key in signature:
            signature[key].append(sig_values)
        else:
            signature[key] = []
            signature[key] += sig_values
    return signature


def results(data, signatures):
    title = file_name.split(".")[0] + "_Results.txt"
    if not os.path.exists(os.path.dirname(title)):
        os.makedirs(os.path.dirname(title))
    f = open(title, "w")
    f.write(title + "\n")
    f.write("\n")
    totalBugs = 0
    totalConformers = 0

    for cluster_index, clustered_namedStructure in enumerate(data):
        numBugs = 0
        numConformers = 0

        cluster_signature = signatures[cluster_index]

        for namedStructure in clustered_namedStructure:
            keys = extract_keys(namedStructure[0])
            namedStructure = flatten_json((namedStructure[0]), '=')
            isNamedStructureABug = False

            newNamedStructure = {}
            for key, value in namedStructure.items():
                flag = 0
                for index, char in enumerate(key):
                    if char == '0' or char == '1' or char == '2' or char == '3' or char == '4' or char == '5' or char == '6' or char == '7' or char == '8' or char == '9':
                        flag = 1
                        if index == len(key)-1:
                            new_key = str(key[0:index-1])
                            newNamedStructure[new_key] = value
                        else:
                            new_key = str(key[0:index-1]) + str(key[index+1:len(key)])
                            newNamedStructure[new_key] = value
                if not flag:
                    newNamedStructure[key] = value
                    flag = 0

            for propertyKey, propertyValue in newNamedStructure.items():
                try:
                    propValues = cluster_signature[propertyKey]
                except:
                    print("EXCEPTION OCCURRED!")
                    print(propertyKey)
                for value in propValues:
                    if value[0] == propertyValue and value[1] == '!':
                        numBugs += 1
                        isNamedStructureABug = True

            if isNamedStructureABug:
                numBugs += 1
            else:
                numConformers += 1

        numBugs = len(clustered_namedStructure) - numConformers
        f.write("Cluster Index: " + str(cluster_index) + "\n")
        f.write("   Number of elements in Cluster = " + str(len(clustered_namedStructure)) + "\n")
        f.write("   Number of Bugs using Z-score: " + str(len(clustered_namedStructure) - numConformers) + "\n")
        f.write("   Number of Conformers using Z-score: " + str(numConformers) + "\n")
        f.write("\n")
        totalBugs += numBugs
        totalConformers += numConformers
    print("Total Bugs = ", totalBugs)
    print("Total Confomers = ", totalConformers)
    f.write("\n")
    f.write("\n")
    f.write("Total Bugs using Z-score: " + str(totalBugs) + "\n")
    f.write("Total Conformers using Z-score: " + str(totalConformers))


def transform_data(data):
    """
    A helper function to extract nested keys from the ACL and to add the frequency of the repeated value. Helps score data.
    Input:
        An ACL in the form {key1:value1, key2:{key3:value3}, key4:[value4], key5:[key6:{key7:value7}]}.
    Output:
        Extracted nested keys from the extract_keys function along with the frequency count.
        Example:
            [
            {key1:{key2:value1, key3:value2, key4:{key5:value3}}
            {key6:{key7:value2}
            {key8:{key3:value3, key4:value5, key6:value3}}
            ]
             Returns a new array with the nested keys appended along with a tuple containing the unnested value along with the frequency count.
            [{
              key1=key2:{'value1':1},
              key1=key3:{'value2':2},
              key1=key4=key5:{'value3':3},
              key6=key7:{'value2':2},
              key8=key3:{'value3':3},
              key8=key4:{'value5':1},
              key8=key6:{'value3':3}
            }]
    """
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
                        try:
                            new_value = new_value[i]
                        except:
                            new_value = new_value[-1]
                    elif (type(new_value) == list) and (len(new_value) == 1):
                        new_value = new_value[0]
                    value = new_value

                if element not in overall:
                    overall[element] = {}

                if type(value) != dict and type(value) != list:
                    if value not in overall[element]:
                        overall[element][value] = 1
        i += 1
    return overall


def calculate_signature_score(signature):
    """
    Calculates the signature score for each signature as the sum of all the weights in it but ignoring the weights marked with "*".
    Input:
        A signature that contains tags of whether or not the weight should be included in calculating the signature.
    Output:
        An array containing the weights of all the signatures that should be considered.
        Example:
        Consider [
        {'key1=key2':['val1', 40], 'key3=key4':['val2':90]}, //40 + 90
        {'key5=key6=key7':['val3', *, 20], 'key8=key9':['val4':80]}, //80
        {'key10=key11':['val5', 40]}  //40

        Returns [130, 80, 40].
    """
    score_arr = []

    for sig in signature:
        score = 0
        for key, value in sig.items():
            for val in value:
                if (val[1] != "!") and (val[1] != "*"):
                    score += val[1]
                elif val[1] == "!":
                    score += val[2]
        score_arr.append(score)

    return score_arr


def calculate_namedstructure_scores(data_final, all_signatures):
    """
    Calculate the individual scores for each discrete-ACL. This includes calculating human_error scores,
    signature_scores, and deviant scores.
    Input:
        data_final:
            List of ACLs grouped into a Cluster.
            Example:
                [
                    [acl-1, acl-4, acl-5, acl-9], //Cluster-0
                    [acl-2, acl-3], //Cluster-1
                    [acl-7], //Cluster-2
                    [acl-6, acl-8] //Cluster-3
                ]
        all_signatures:
            Consolidated signature for each Cluster.
    Output:
        deviant_arr: Returns all deviant properties for the ACL. Empty list is returned if no deviant property
        in the ACL.
        count_arr: [[TODO]]
        dev_score: Returns the deviant score for the deviant properties found. 0 if no deviant property.
        acls_arr: [[TODO]]
        sig_score: Returns the signature score of the ACL.
        cluster_num: Returns the cluster number that the ACL belongs to.
        acls_score: The score that is generated for each acl
        human_errors_arr: Returns the human_error properties (IPValidity, DigitRepetition, PortRange) for each ACL and
        empty list if no human_error properties present in the ACL.
        human_error_score: Returns the score of the human error property calculated for the ACL. 0 is returned if
        no human_error property exists in the ACL.
    """

    deviant_arr = []
    count_arr = []
    acls_dict = {}
    acls_arr = []
    acls_score = []
    sig_score = []
    dev_score = []
    cluster_num = []
    human_errors_arr = []
    human_errors_score = []
    i = 0


    for acl_list in data_final:
        bug_count = 0
        conformer_count = 0
        signature = all_signatures[i]
        for acl in acl_list:
            flag = 0
            if str(acl[0]) not in acls_dict:
                acls_dict[str(acl[0])] = 1
                acls_arr.append(acl[0])
                cluster_num.append(i)
                flag = 1
            else:
                print(acl[0])
                print(acls_dict)
                continue
            sig_score.append(signature_scores[i])
            deviant = []
            count = 0
            dev_c = 0
            acl_c = 0
            human_errors = []
            human_error_category = {}
            data = transform_data(acl)
            for data_key, data_val in data.items():
                if data_key in signature:

                    '''
                    Key Valid. Now check for actual Value
                    '''
                    for val in data_val.items():
                        (error_key, error_value), error_category = calculateHumanErrors(data_key, val[0], signature[data_key], file_name.split(".")[0])
                        if error_category:
                            human_errors.append((error_key, error_value))
                            if error_category not in human_error_category:
                                human_error_category[error_category] = 0
                            human_error_category[error_category] += 1
                        for sig_val in signature[data_key]:
                            if val[0] == sig_val[0]:

                                '''
                                value also present. Now check if value part of bug/sig/skip
                                '''
                                if sig_val[1] == "!":
                                    dev_c += sig_val[2]
                                    acl_c += sig_val[2]
                                    deviant.append((data_key, sig_val[0]))
                                    bug_count += 1
                                elif sig_val[1] == "*":
                                    conformer_count += 1
                                    continue
                                else:
                                    conformer_count += 1
                                    count += sig_val[1]
                                    acl_c += sig_val[1]
                else:

                    '''
                    Deviant Key
                    '''
                    if data_key != "lines=name":
                        deviant.append(data_key)
                        dev_c += data_val
                        acl_c += data_val

            if flag == 1:
                count_arr.append(count)
                deviant_arr.append(deviant)
                dev_score.append(dev_c)
                acls_score.append(acl_c)
                human_errors_arr.append(human_errors)
                human_errors_score.append(calculate_human_error_score(human_error_category))
        i += 1
    return deviant_arr, count_arr, dev_score, acls_arr, sig_score, cluster_num, acls_score, human_errors_arr, human_errors_score


def checkIPValidity(ip_address):
    """
    A reg-ex check to verify the validity of an IP address.
    Input:
        A list of IP addresses
    Output:
        A boolean representing the validity of the IP address.
        Returns 'True' if all the IPs are valid and 'False' if any of the IP is invalid.
    """
    try:
        ip_address = ip_address.split(":")
        for ip in ip_address:
            IP_check = "^(([0-9]|[1-9][0-9]|1[0-9]{2}|2[0-4][0-9]|25[0-5])\.){3}([0-9]|[1-9][0-9]|1[0-9]{2}|2[0-4][0-9]|25[0-5])?(\/)?((3[01]|3[02]|[12][0-9]|[0-9])?)$"
            match = re.match(IP_check, ip)

            if not match:
                return False
        return True
    except e:
        print(e)
        return True


def checkPortRange(port_range):
    """
    A check to verify that the port range is specified correctly (elem0 <= elem1).
    Input:
        A string that contains two numbers separated by a '-'.
    Output:
        A boolean representing the validity of the range (elem0 <= elem1).
        Example:
            52108-52109 (True)
            466 - 466 (True)
            466 - 465 (False)
    """

    try:
        port_split = port_range.split("-")
        if port_split[-1] < port_split[0]:
            return False
        return True
    except:
        return True


def checkDigitRepetition(digit, signature):
    """
    Checks for Digit repetition.
    Input:
        The value for the following keys: srcPorts, dstPorts, lengthRange
    Output:
        Returns True if there is any Human Error and the digit is repeated twice.
    """

    try:
        if type(digit) == str:
            digit = float(digit.split(":")[0])
        if digit == 0:
            return False
        for item in signature:
            if type(item) == str:
                item = int(item.split(":")[0])
            if digit == (item*10+item%10):
                print("--------", digit, item*10 + item%10)
                return True
        return False
    except:
        return False


def calculateHumanErrors(data_key, data, signature, namedStructure):
    """
    Checks for simple human errors like entering invalid IP Addresses, incorrect port-ranges, and digit repetitions.
    Input:
        data_key: The nested keys calculated in the overall_dict and get_overall_dict methods.
        Example: key1=key2=key4
        data: The data value for the keys.
        signature: The signature for the keys that was calculated in the calculate_signature_d method.
        namedStructure: The type of the IP file.
        Possible values: IP_Access_List, Route_Filter_List, Routing_Policy, VRF, others.
    Output:
        Returns the error and the category it belongs to.
        Example:
            key1=key2=key3 [1333.0.0.13] [1333.0.0.13] IP_Access_List
            Returns:
                key1=key2=key3 [1333.0.0.13] IP
    """
    human_error = (None, None)
    category = None
    data_key = data_key.split("=")[-1]
    signature_items = []
    for sig_item in signature:
        signature_items.append(sig_item[0])

    if namedStructure == "IP_Access_List":
        if data_key == "ipWildcard":
            if not checkIPValidity(data):

                '''
                Invalid IP
                '''
                human_error = (data_key, data)
                category = "IP"
        elif data_key in ["dstPorts", "srcPorts"]:
            if not checkPortRange(data):

                '''
                Invalid Ports Range
                '''
                human_error = (data_key, data)
                category = "RANGE"

    elif namedStructure == "Route_Filter_List":
        if data_key == "ipWildcard":
            if not checkIPValidity(data):

                '''
                Invalid IP
                '''
                human_error = (data_key, data)
                category = "IP"
        elif data_key == "lengthRange":
            if not checkPortRange(data):

                '''
                Invalid Ports Range
                '''
                human_error = (data_key, data)
                category = "RANGE"

    elif namedStructure == "Routing_Policy":
        if data_key == "communities":
            if checkDigitRepetition(data, signature_items):

                '''
                Error Copying digits
                '''
                human_error = (data_key, data)
                category = "DIGIT"
        elif data_key == "ips":
            if not checkIPValidity(data):

                '''
                Invalid IP
                '''
                human_error = (data_key, data)
                category = "IP"

    elif namedStructure == "VRF":
        if data_key in ["administrativeCost", "remoteAs", "metric", "localAs", "referenceBandwidth", ]:
            if checkDigitRepetition(data, signature_items):

                '''
                Error Copying digits
                '''
                human_error = (data_key, data)
                category = "DIGIT"
        elif data_key in ["peerAddress", "localIp", "routerId", "network"]:
            if not checkIPValidity(data):

                '''
                Invalid IP
                '''
                human_error = (data_key, data)
                category = "IP"

        '''
        Any Other namedStructure
        '''
    else:
        try:
            if re.search('IP|ip', data_key) and not re.search('[a-zA-Z]', data):
                if not checkIPValidity(data):

                    '''
                    Invalid IP
                    '''
                    human_error = (data_key, data)
                    category = "IP"
            elif not re.search("[a-zA-Z]", data):
                if checkDigitRepetition(data, signature_items):

                    '''
                    Error Copying digits
                    '''
                    human_error = (data_key, data)
                    category = "DIGIT"
        except:
            pass

    return human_error, category


def calculate_human_error_score(category_dict):
    """
    Scores the human_errors that have been found with IPValidity and DigitRepetition errors
    weighed as 'high,' i.e, 0.8 and PortRange errors weighed 'medium,' i.e., 0.5.
    Input:
        A dictionary containing the count of the error occurrences.
    Output:
        A weighted sum of all the errors found.
    """

    total_score = 0
    low = 0.2
    medium = 0.5
    high = 0.8
    weightage_dict = {"IP": high, "RANGE": medium, "DIGIT": high}
    for category, count in category_dict.items():
        if count != 0:
            #print("* Human Error Found *")
            total_score += weightage_dict[category]/np.log(1+count)

    return round(total_score/len(category_dict), 2) if category_dict else total_score


def flatten_json(data, delimiter):
    """
    Flattens a JSON file.
    Input:
        data:
            A JSON dictionary of hierarchical format.
            {key1: {key2: value2, key3: value3}, key4: {key5: value5, key6: [value6, value7, value8]}}
        delimiter:
            A parameter to separate the keys in order to facilitate easy splitting.
    Output:
        A flattened dictionary with keys separated by the delimiter parameter.
        key1_key2:value2, key1_key3:value3, key4_key5:value5, key4_key6:value6, key4_key6:value7, key4_key6:value8
    """

    out = {}

    def flatten(data, name=''):
        if type(data) is dict:
            for key in data:
                flatten(data[key], name + key + delimiter)
        elif type(data) is list:
            i = 0
            for elem in data:
                flatten(elem, name + str(i) + delimiter)
                i += 1
        else:
            out[name[:-1]] = data

    flatten(data)
    return out


def encode_data(data):
    """
    Converts categorical values into numeric values. We use MultiLabelBinarizer to encode categorical data.
    This is done in order to pass the data into clustering and other similar algorithms that can only handle numerical data.
    Flattens each ACL list and then encodes them.
    Input:
        A Python list that contains all discrete-ACLs.
    Output:
        A Python list after encoding.
    """

    flattenedData = []
    allKeys = []
    for NS in data:
        flattenedNamedStructure = flatten_json(NS, '_')
        flattenedData.append(flattenedNamedStructure)

        for key in flattenedNamedStructure.keys():
            if key not in allKeys:
                allKeys.append(key)

    mergedData = []
    for NS in flattenedData:
        mergedNS = []
        for key, value in NS.items():
            mergedNS.append(str(value))
        mergedData.append(mergedNS)
    mlb = MultiLabelBinarizer()

    data_T = mlb.fit_transform(mergedData)
    print("MLb classes=")
    print(mlb.classes_)
    return data_T, mlb.classes_


def export_clusters(data, acl_weight_mapper):
    """
        Helper Method to verify authenticity of Clusters being formed.
        Input:
            The data that is sorted into list of Clusters.
            Example:
                [
                    [acl-1, acl-4, acl-5, acl-9], //Cluster-0
                    [acl-2, acl-3], //Cluster-1
                    [acl-7], //Cluster-2
                    [acl-6, acl-8] //Cluster-3
                ]
            We also make use of acl_dict and node_name_dict dictionaries by searching for the ACL and
            then getting the appropriate ACL_name and the nodes that the ACL is present in.
        Output:
            A csv file by the name of Generated_Clusters is written in the format:
            Cluster-0  ||||  Cluster-0 Names      |||| Cluster-0 Nodes              |||| Cluster-1 |||| Cluster-1 Names          |||| Cluster-1                 Nodes
            acl-1      ||||  permit tcp eq 51107  |||| st55in15hras                 |||| acl-2     |||| permit udp any eq 1200   |||| rt73ve11m5ar
            acl-4      ||||  permit tcp eq 51102  |||| st55in15hras, st55in17hras   |||| acl-3     |||| permit udp any eq 120002 |||| rt73ve10m4ar
            acl-5      ||||  permit tcp eq 51100  |||| st55in17hras                 ||||
            acl-9      ||||  permit tcp eq 51109  |||| st55in17hras                 ||||
    """

    column_labels = []

    for index in range(len(data)):
        column_labels.append("Cluster " + str(index))
        column_labels.append("Cluster " + str(index) + " ACL Weights")
        column_labels.append("Cluster " + str(index) + " Nodes")

    data_to_export = pd.DataFrame(columns=column_labels)
    for cluster_index, cluster_data in enumerate(data):
        discrete_ACL_nodes = []
        cluster_weights = []
        for discrete_ACL in cluster_data:
            temp = json.dumps(discrete_ACL[0], sort_keys=True)
            temp_arr = []
            try:
                for node in namedstructure_node_mapper[temp]:
                    temp_arr.append(node)
                discrete_ACL_nodes.append(temp_arr)
            except:
                discrete_ACL_nodes.append(None)
            cluster_weights.append(acl_weight_mapper[temp])
        cluster_data = pd.Series(cluster_data)
        cluster_weights_series = pd.Series(cluster_weights)
        discrete_ACL_nodes = pd.Series(discrete_ACL_nodes)
        data_to_export["Cluster " + str(cluster_index)] = cluster_data
        data_to_export["Cluster " + str(cluster_index) + " ACL Weights"] = cluster_weights_series
        data_to_export["Cluster " + str(cluster_index) + " Nodes"] = discrete_ACL_nodes

    file = file_name.split(".")[0]
    print(file)
    title = "Clusters_" + file + ".csv"
    print(title)
    data_to_export.to_csv(title)


def parse_data():
    """
    A helper method to parse through the input configuration files and capture necessary information.
    Input:
        None. The file path parameter is read from the commandline arguments.
    Output:
        discrete_namedstructure: A list that contains stringified named-structures.
        namedstructure_nod_mapper: A dictionary that contains the named-structure configuration as key and a list of
        nodes it is a part of as value.
    """
    df = pd.read_json(sys.argv[2], orient="index")

    discrete_namedstructure = []
    namedstructure_node_mapper = {}  # Maps each discrete_acl with all the nodes that it belongs to
    discrete_nodes = []

    for column in df.columns:
        for index, data in df[column].iteritems():
            if data is not None:
                if 'lines' in data[0]:
                    data_holder = 'lines'
                    data_to_look_under = data[0][data_holder]
                elif 'statements' in data[0]:
                    data_holder = 'statements'
                    data_to_look_under = data[0][data_holder]
                else:
                    data_to_look_under = data

                for discrete_acl in data_to_look_under:
                    if 'name' in discrete_acl:
                        del discrete_acl['name']

                    discrete_acl = json.dumps(discrete_acl, sort_keys=True)
                    discrete_namedstructure.append(discrete_acl)

                    if discrete_acl in namedstructure_node_mapper:
                        nodes = namedstructure_node_mapper[discrete_acl]
                        if index not in nodes:
                            nodes.append(index)
                        namedstructure_node_mapper[discrete_acl] = nodes
                    else:
                        namedstructure_node_mapper[discrete_acl] = [index]

                if index not in discrete_nodes:
                    discrete_nodes.append(index)

    print("The number of discrete nodes in a network is: ", len(discrete_nodes))
    return discrete_namedstructure, namedstructure_node_mapper


def perform_pca_analysis(encoded_data, column_names):
    """
        A helper method to analyse the data using PCA
    """

    pca = PCA()
    pca.fit(encoded_data)

    cumulative_variance = np.cumsum(np.round(pca.explained_variance_ratio_, decimals=8) * 100);
    labels = [x for x in range(1, len(cumulative_variance) + 1)];
    loadings = pd.DataFrame(pca.components_.T, columns=labels, index=column_names)

    significance = {}

    for index in loadings.index:
        temp_list = loadings.loc[index]
        sig = 0
        for value in temp_list:
            sig += value * value
        significance[index] = sig

    plt.plot(cumulative_variance)
    plt.xlabel("N-components")
    plt.ylabel("Cumulative Explained Variance")
    plt.show()

    sorted_significance = sorted(significance.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)

    top_ten_attributes = []

    for sigAttr in sorted_significance:
        top_ten_attributes.append(sigAttr[0])
    print("Top Ten Attributes:")
    print(top_ten_attributes)


def silhouette_analysis(data, acl_weights):
    """
        A helper method to perform an analysis of various scoring functions
    """
    from sklearn.metrics import silhouette_score, davies_bouldin_score

    k_range = range(2, 30)

    elbow_scores = []
    silhouette_scores = []
    davies_bouldin_scores = []

    elbow_file = open("elbow_scores.txt", "w")
    silhouette_file = open("silhouette_scores.txt", "w")
    davies_bouldin_file = open("davies_bouldin_scores.txt", "w")

    for num_clusters in k_range:
        print(num_clusters)
        kmeans = KMeans(n_clusters=num_clusters)
        cluster_labels = kmeans.fit_predict(data, None, sample_weight=acl_weights)

        cluster_centers = kmeans.cluster_centers_
        k_distance = cdist(data, cluster_centers, "euclidean")
        distance = np.min(k_distance, axis=1)
        distortion = np.sum(distance) / data.shape[0]

        silhouette_avg = silhouette_score(data, cluster_labels)
        davies_bouldin_avg = davies_bouldin_score(data, cluster_labels)

        silhouette_scores.append(silhouette_avg)
        davies_bouldin_scores.append(davies_bouldin_avg)
        elbow_scores.append(distortion)

        silhouette_file.write(str(silhouette_avg) + " ")
        davies_bouldin_file.write(str(davies_bouldin_avg) + " ")
        elbow_file.write(str(distortion) + " ")

    kn_elbow = KneeLocator(list(k_range), elbow_scores, S=5.0, curve='convex', direction='decreasing')
    plt.scatter(x=k_range, y=elbow_scores)
    plt.xlabel("Range")
    plt.ylabel("Elbow Score")
    plt.vlines(kn_elbow.knee, plt.ylim()[0], plt.ylim()[1], linestyles='dashed')
    plt.show()

    kn_silhouette = KneeLocator(list(k_range), silhouette_scores, S=5.0, curve='convex', direction='increasing')
    plt.scatter(x=k_range, y=silhouette_scores)
    plt.xlabel("Range")
    plt.ylabel("Silhouette Score")
    plt.vlines(kn_silhouette.knee, plt.ylim()[0], plt.ylim()[1], linestyles='dashed')
    plt.show()

    kn_davies_bouldin = KneeLocator(list(k_range), davies_bouldin_scores, S=5.0, curve='convex', direction='decreasing')
    plt.scatter(x=k_range, y=davies_bouldin_scores)
    plt.xlabel("Range")
    plt.ylabel("Davies Bouldin Score")
    plt.vlines(kn_davies_bouldin.knee, plt.ylim()[0], plt.ylim()[1], linestyles='dashed')
    plt.show()


'''
# Helper Methods End 
# *****************************************************************************
# *****************************************************************************
'''

whitelistDict = {}
Z_SCORE_FLAG = 0
ACTION_FLAG = 0
k_select = 0

'''
Parsing Data
'''

try:
    file_name = sys.argv[2].split("/")[-1]
    network_name = "DATA_HERE_" + sys.argv[2].split("/")[-2]
    print(network_name)
    '''
    Making Outlier Directory for Current Network
    '''
    if not os.path.exists(network_name):
        os.makedirs(network_name)
    flag_file = network_name + '/' + '.flag_' + file_name
    if sys.argv[1] == "-j":
        df = pd.read_json(sys.argv[2], orient="index")
        try:
            if sys.argv[4] == "-a":
                ACTION_FLAG = 3
        except:
            ACTION_FLAG = 0

        try:
            Z_SCORE_FLAG = int(sys.argv[3])
        except:
            error_msg("Invalid Z-Score Argument sent", sys.argv[3])

        f = open(flag_file,'w')
        f.write('{}'.format(ACTION_FLAG))
        f.close()

    elif sys.argv[1] == "-e":
        df = pd.read_json(sys.argv[2], orient= "index")
        try:
            with open(sys.argv[3], 'rb') as handle:
                whitelistDict = pickle.load(handle)
        except:
            print("FileNotFoundError: Please check if file exists.")
        ACTION_FLAG = 1
    elif sys.argv[1] == "-d":
        df = pd.read_json(sys.argv[2],orient = "index")
        ACTION_FLAG = 2
    else:
        error_msg("Invalid Argument or flags sent", sys.argv[1])

except:
    error_msg("Invalid File specified. Please check the input dataset", sys.argv[2])


outlier_filename = network_name + '/' + 'outlier_' + file_name
cluster_filename = network_name + '/' + '.cluster_' + file_name
sig_filename = network_name + '/' + '.sig_' + file_name
outlier_nodes_filename = network_name + '/' + '.outlier_nodes_' + file_name
print(outlier_filename, cluster_filename ,sig_filename, outlier_nodes_filename)
print("===========================================================")
print(Fore.BLUE, end='')
print("outlier-analyzer code started ...")
print(Style.RESET_ALL)
print(Fore.GREEN, end='')


start = time.time()

'''
    Calculating outliers selected
'''

f = open(flag_file, 'r')
flag = f.readline()
f.close()

discrete_namedstructure, namedstructure_node_mapper = parse_data()


if (ACTION_FLAG == 0) or (ACTION_FLAG == 3):

    mlb = MultiLabelBinarizer()
    ns_weight_mapper = {}
    data_for_clustering = []
    namedstructure_weights = []

    for ns in discrete_namedstructure:
        if ns not in ns_weight_mapper.keys():
            ns_weight_mapper[ns] = 1

        else:
            value = ns_weight_mapper[ns]
            ns_weight_mapper[ns] += 1

    for ns, weight in ns_weight_mapper.items():
        ns = json.loads(ns)
        data_for_clustering.append(ns)
        namedstructure_weights.append(weight)

    encodedLists, column_names = encode_data(data_for_clustering)
    df_enc = pd.DataFrame(encodedLists)
    df_enc = df_enc.dropna(axis=1, how='any')

    # perform_pca_analysis(encodedLists, column_names)

    print("data encoding done...")

    '''
       Perform K-Means
    '''
    print("starting data clustering...")
    perform_kmeans_clustering(df_enc, namedstructure_weights)
    print("data clustering done...")

    # silhouette_analysis(df_enc, acl_weights)

    '''
    Grouping data based on their Clusters
    '''
    cluster_range = np.arange(k_select)
    data_final = []
    data_final_enc = []
    for index in cluster_range:
        temp = []
        temp_enc = []
        for i in range(len(df_enc)):
            if df_enc['kmeans_cluster_number'][i] == index:
                temp.append([data_for_clustering[i]])
                temp_enc.append([data_for_clustering[i]])
        data_final.append(temp)
        data_final_enc.append(temp_enc)

    # export_clusters(data_final, acl_weight_mapper)
    '''
    Writing Clustered Data into a file
    '''
    with open(cluster_filename, 'w') as f:
        f.write(json.dumps(data_final))

    '''
    Calculating Overall Structure per Cluster
    '''
    if ACTION_FLAG == 3:
        overall_array_0 = overall_dict(data_final)
    try:
        overall_array = get_overall_dict(data_final)
    except:
        overall_array = overall_dict(data_final)

    '''
    Generating Signatures
    '''
    all_signatures = []
    for i in range(len(overall_array)):
        signature = calculate_signature_d(overall_array[i])
        all_signatures.append(signature)
    print("signature creation done...")

    '''
    Retuning Signature
    '''
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

    all_signatures = all_signatures[0]
    wlDict = copy.deepcopy(whitelistDict['deviant'])
    for edit_key, edit_value in whitelistDict['deviant']:
        flag = 0
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
        if flag == 1:
            wlDict.remove((edit_key, edit_value))

    if wlDict:
        print(Fore.RED, end='')
        print("\nERROR : Specified Attributes {} either\n\tnot present or not a bug!".format(wlDict))
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

    '''
    Displaying the Outlier Nodes
    '''
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

    print(Style.RESET_ALL)
    print("########################")
    print("Outlier Nodes are:")
    outlier_nodes_arr = outlier_nodes_arr[0]
    print(Fore.RED, end='')
    print(*outlier_nodes_arr, sep="\n")
    print(Style.RESET_ALL)
    print("########################")
    sys.exit(0)


'''
Scoring Signature
'''
signature_scores = calculate_signature_score(all_signatures)
print("signature scoring done...")

'''
Scoring ACLs
'''
deviant_arr, count_arr, dev_score, acls_arr, sig_score, cluster_num, acls_score, human_errors_arr, human_errors_score \
    = calculate_namedstructure_scores(data_final, all_signatures)

print("acl scoring done...")

'''
Calculate outlier nodes
'''
count = 0
outlier_nodes = set()
for i in range(len(deviant_arr)):
    if len(deviant_arr[i]) > 0:
        count += 1
        temp = json.dumps(acls_arr[i], sort_keys=True)
        for item in namedstructure_node_mapper[temp]:
            outlier_nodes.add(item)

with open(outlier_nodes_filename, 'w') as f:
    f.write(json.dumps(list(outlier_nodes)))

'''
writing all signature to a hidden file
'''
with open(sig_filename, 'w') as f:
    f.write(json.dumps(all_signatures))

nodes = []

for i in range(len(acls_arr)):
    temp = json.dumps(acls_arr[i], sort_keys=True)
    tempArr = []
    try:
        for item in namedstructure_node_mapper[temp]:
            tempArr.append(item)
        nodes.append(tempArr)
    except:
        nodes.append(None)


'''
    Creating dataframe and exporting as a json file
'''

df_final = pd.DataFrame()
with open("deviant_array.txt", "w") as f:
    print(deviant_arr, file=f)

print(human_errors_arr)

master_signatures = []

for i in range(len(data_final)):
    for index in data_final[i]:
        master_signatures.append(all_signatures[i])



# df_final['acl_name'] = acl_names
df_final['cluster_number'] = cluster_num
df_final['Conformer/Signature Definition'] = master_signatures
df_final['acl_structure'] = acls_arr
df_final['nodes'] = nodes
df_final['deviant_properties'] = deviant_arr
df_final['human_error_properties'] = human_errors_arr
df_final['human_error_score'] = human_errors_score
df_final['similarity_score'] = count_arr
df_final['acl_score'] = acls_score
df_final['max_sig_score'] = sig_score

outlier_flag = ['T' if len(deviant_prop)==0 else 'F' for deviant_prop in deviant_arr]
df_final['Outlier Flag'] = outlier_flag

df_final.to_json(outlier_filename, orient='split', index=False)
print(Style.RESET_ALL, end="")
end = time.time()
print(df_final)


print("###")
print(Fore.BLUE, end='')
print("OUTLIER-ANALYZER SUCCESSFUL #")
print("time to run : {} seconds".format(round(end - start), 3))
print(Style.RESET_ALL, end='')
print()

print("###########################################################")
print(outlier_nodes)
print(Fore.BLUE, end='')
print("\nTotal Outliers Count = {}".format(len(outlier_nodes)))
print(Style.RESET_ALL, end='')
print("\nTo view the detailed report, open the")
print("json file named: '{}'\n".format(outlier_filename))
print("###########################################################")

print()

sys.exit(0)

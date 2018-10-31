import json
import re
from colorama import Fore, Back, Style


def extract_keys(the_dict, prefix=''):
    # TODO
    # fix bug with list of dicts not being extracted
    # but only first element

    key_list = []
    
    for key, value in the_dict.items():
    
        # set the prefix
        if len(prefix) == 0:
            new_prefix = key 
        else:
            new_prefix = prefix + '.' + key 

        # recursively call extract_keys for nested dicts 
        if type(value) == dict:
            key_list.extend(extract_keys(value, new_prefix))
        elif type(value) == list and type(value[0]) == dict:
            key_list.extend(extract_keys(value[0], new_prefix))
        else:
            key_list.append(new_prefix)


    return key_list



overall = {}



with open('input.json') as json_data:
    datas = json.load(json_data)



for data in datas:

    result = extract_keys(data)
    # print(result)

    for element in result:

        value = data
        for key in element.split('.'):

            new_value = value[key]
            if type(new_value) == list:
                new_value = new_value[0]

            value = new_value

        # print(element, value)
        if element not in overall: 
            overall[element] = [value]
        else:
            overall[element].append(value)

exclude_list = ["name", "lines.matchCondition.ipWildCard"]
exclude_list = []


def numberRange(value):


    searchObj = re.search('(\d+)-(\d+)', value)
    if searchObj:
        lower = int(searchObj.group(1))
        upper = int(searchObj.group(2))

        num_range = list(range(lower, upper+1))

        for i in range(len(num_range)):
            num_range[i] = str(num_range[i])

        return set(num_range)

    else:
        return None


def isEqual(index1, index2):

    print(index1, " vs ", index2, "\n")

    isEqual = True

    for key, value in overall.items():

        if key not in exclude_list:

            value1 = numberRange(value[index1])
            if value1 == None:
                value1 = set([value[index1]])

            value2 = numberRange(value[index2])
            if value2 == None:
                value2 = set([value[index2]])

            intersect = value1.intersection(value2)
            
            if len(intersect) == 0:
                isEqual = False
                print(Fore.RED)


            print("Key:", key)
            print("Value 1:", value1)
            print("Value 2:", value2)

            print(Style.RESET_ALL)

    print(isEqual)
    print("\n=========================\n")
            

    return isEqual


result = isEqual(0, 1)


result = isEqual(1, 2)

result = isEqual(1, 1)










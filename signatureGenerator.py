import re
import json
from colorama import Fore, Back, Style
import sys
import pickle

# this file generates a signature and saves it to a file

props = []
datas = []

# f = open('datasets/flat-sample/namedStructureProperties_ip-accesslist.json')
f = open(sys.argv[1])

# selected features inputted as command line argument
selection = sys.argv[2].split(',')

# go through each line of the file (which represents a feature)
# and add each feature that corresponds to a selection
count = 0
for line in f:

    match = re.match('.*?:(.*)=>(.*);', line)

    try:
        prop = match.group(1)
        props.append(prop)

        extracted = match.group(2)
        extracted = '[' + extracted + ']'
        data = json.loads(extracted)

    except AttributeError:
        pass

    for i in range(len(data)):
        data[i] = [data[i]]

    if prop in selection:
        datas.append(data)

    count += 1
    if count > max(selection):
        break


# go through the dict and recursively build up a list of fields in the dict
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

# overall represents a mapping of fields to list of values (over all the entries)
overall = {}

# for each feature/column, go through each row of that feature
# and extract the key/values recursively using the extract_keys() function
# and add them to the overall dict.
for data in datas:

    for item in data:

        result = extract_keys(item[0])
        # print(result)

        for element in result:

            value = item[0]
            for key in element.split('.'):

                new_value = value[key]
                if type(new_value) == list:
                    new_value = new_value[0]

                value = new_value

            # print(element, value)
            if element not in overall:
                # overall[element] = [value]
                overall[element] = {}

            if value not in overall[element]:
                overall[element][value] = 1
            else:
                overall[element][value] += 1


# Create signature by going through the overall dict
# and for each field, picking the most common value.
# The weight is determined by seeing what percentage
# of the total list of values the most common value is.

signature = {}

for key, value in overall.items():
    # print(key, value)
    max = 0
    sum = 0
    most = None
    for k, v in value.items():
        sum += v
        if v > max:
            max = v
            most = k
    weight = int(max / sum * 100)

    signature[key] = (most, weight)

#     print(most)
#     print(weight)
#     print()
#


# write signature to file
pickle.dump(signature, open('signature.txt', 'wb'))


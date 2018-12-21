import re
import json
from colorama import Fore, Back, Style
import sys
import pickle

props = []
datas = []

f = open('datasets/flat-sample/namedStructureProperties_ip-accesslist.json')

selection = [1, 5, 10]

count = 0
for line in f:

    match = re.match('.*?:(.*)=>(.*);', line)

    try:
        props.append(match.group(1))

        extracted = match.group(2)
        extracted = '[' + extracted + ']'
        data = json.loads(extracted)

    except AttributeError:
        pass

    for i in range(len(data)):
        # data[i] = str(data[i])
        data[i] = [data[i]]

    if count in selection:
        datas.append(data)

    count += 1
    if count > max(selection):
        break


def isHomogeneous(input_dict):

    values = list(input_dict.values())

    print(values)

    sum = 0
    max = 0
    for value in values:
        if value > max:
            max = value
        sum += value

    ratio = max / sum
    print(ratio)

    if ratio < 0.7:
        return False
    else:
        return True


# This function re
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

excluded = []

for key, value in overall.items():
    if isHomogeneous(value):
        print(Fore.GREEN + key, ": ", value)
    else:
        print(Fore.RED + key, ": ", value)
        excluded.append(key)
    print(Style.RESET_ALL)
    print()
print()

print('Excluded:', excluded)



signature = pickle.load(open('signature.txt', 'rb'))


print()
for key, value in overall.items():
    print(key, ':', value)
    print()

print(Fore.MAGENTA)
print(signature)
print(Style.RESET_ALL)

# compare to signature

for i, item in enumerate(datas[0]):

    match = 0
    total = 0

    print('=' * 141, end='\n\n')
    print("Entry #%d" % i, end='\n\n')

    for data in datas:

        item = data[i]

        print(item, end='\n\n')

        for key, value in signature.items():

            if key in excluded:
                continue

            current = item[0]

            key_list = key.split('.')

            for k in key_list:
                # print(k)
                if k in current:
                    current = current[k]
                    if type(current) == list:
                        current = current[0]
                else:
                    break

            if type(current) == dict:
                continue

            print(key)

            if current == value[0]:
                match += value[1]
                print(Fore.BLUE, end='')
            else:
                print(Fore.RED, end='')

            print('Entry value: ', current)
            print('Signature value: ', value[0])
            print(Style.RESET_ALL)

            total += value[1]

    print(Fore.GREEN, end='')
    print(match, '/', total)
    print(Style.RESET_ALL)

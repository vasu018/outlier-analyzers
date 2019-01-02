import re
import json
from colorama import Fore, Back, Style

props = []
datas = []

f = open('datasets/flat-sample/namedStructureProperties_ip-accesslist.json')
# f = open('datasets/flat-sample/namedStructureProperties_routepolicies.json')


count = 0
for line in f:
    # print(line)
    # print()

    match = re.match('.*?:(.*)=>(.*);', line)

    props.append(match.group(1))

    extracted = match.group(2)
    extracted = '[' + extracted + ']'

    data = json.loads(extracted)

    for i in range(len(data)):
        # data[i] = str(data[i])
        data[i] = [data[i]]

    if count == 1:
        datas.append(data)

    count += 1
    if count == 2:
        break


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


# for item in datas[0]:
#     key_list = extract_keys(item[0])
#
#     print(item[0])
#     print(key_list)
#     print()


overall = {}

for item in datas[0]:

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



# create signature

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
    print(item, end='\n\n')


    for key, value in signature.items():

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

        # print(current)
        # print()

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


    # print(item)
    # print()
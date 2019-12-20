import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from colorama import Fore, Back, Style


def error_msg():
    print("****************************")
    print(Fore.BLUE, end='')
    print("Invalid file/flags sent!")
    print(Style.RESET_ALL, end='')
    print("****************************")
    sys.exit(0)

ACTION_FLAG = 0
fileName = sys.argv[2].split("/")[-1]
try:
	df = pd.read_json('outlier_'+fileName, orient = "split")
	# For properties
	if sys.argv[1] == "-p":
		ACTION_FLAG = 1
	# For score
	elif sys.argv[1] == "-s":
		ACTION_FLAG = 2
	else:
		error_msg()

except:
	error_msg()

if ACTION_FLAG == 1:
	dev_dict = {}
	dev_prop_dict = {}
	dev_val_dict = {}
	for items in df['deviant_properties']:
	    for prop in items:
	        
	        # For Properties
	        if prop[0] not in dev_prop_dict:
	            dev_prop_dict[prop[0]] = 1
	        else:
	            dev_prop_dict[prop[0]] += 1
	            
	        # For Values
	        if prop[1] not in dev_val_dict:
	            dev_val_dict[prop[1]] = 1
	        else:
	            dev_val_dict[prop[1]] += 1
	        
	        # For as a whole
	        df = prop[0]+" "+prop[1]
	        if df not in dev_dict:
	            dev_dict[df] = 1
	        else:
	            dev_dict[df] += 1

	x, y = zip(*dev_prop_dict.items())
	map_dict = {}
	i = 1
	for item in x:
	    map_dict[item] = i
	    i+=1
	a, b = zip(*map_dict.items())
	print()
	print("Encoding is as follows:")
	print()
	print(Fore.BLUE, end='')
	for i in range(len(a)):
		print("{} => {}".format(b[i], a[i]))
	print(Style.RESET_ALL, end='')
	plt.xlabel("ACL Property Encoded Number")
	plt.ylabel("Frequency of the Deviant Property")
	plt.title("Frequency of the Deviant Propertie's Appearance")
	plt.bar(b, y)
	plt.show()

elif ACTION_FLAG == 2:
	x = np.arange(len(df))
	fig = plt.figure()
	ax = fig.add_subplot(111)

	ax.plot(x, df['similarity_score'])
	ax.plot(x, df['acl_score'] - df['similarity_score'])
	ax.plot(x, df['max_sig_score'])

	plt.legend(['similar', 'deviant', 'signature'], loc=0)

	plt.xlabel("Index number representing an ACL/ACL_List")
	plt.ylabel("Scores")
	plt.title("All Scores Comparision - ACL/ACL_List")
	plt.show()


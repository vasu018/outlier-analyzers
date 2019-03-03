#!/usr/bin/python3
import os, sys, shutil
import datetime
import re
# checking whether path and filename are given.
if len(sys.argv) != 2:
    print "Usage : python rename.py <path>"
    sys.exit()

ext = [".config", ".cfg"]
source_path = sys.argv[1]
source_path = source_path + "/configs/"
print("Source Folder to be anonymized is:", source_path)

def folder_name_generator ():
    folder_name = unicode(datetime.datetime.now())
    folder_name = re.sub(r"\s+", '-', folder_name)
    folder_name = "result_folder_" + folder_name
    return folder_name

main_folder_name = folder_name_generator()
target_folder_path = "./" + main_folder_name + "/configs"
print("Target Folder with anonymized configuration is:", target_folder_path)

os.makedirs( target_folder_path, 0755 );

count =1
try:
    for x in os.walk(source_path):
        for source_file_name in x[2]:
            #if source_file_name.endswith(".cfg"): 
            if source_file_name.endswith(tuple(ext)): 
                source_file_name = os.path.join(source_path,source_file_name)
                target_file_name = "%s/node-%d" %(target_folder_path, count)
                print("Copying %s to %s", source_file_name, target_file_name)
                shutil.copy(source_file_name, target_file_name)
                count = count +1
except OSError:
    pass

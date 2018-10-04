import json
from pprint import pprint

inputFile = "datasets/flat-sample/outputAnonymizedFile.txt"
with open(inputFile) as f:
    data = json.load(f)

for i in range(len(data)):
    print("Node ID:", data[i]['Node']['id'])
    print("Node Name:", data[i]['Node']['name'])
    print("DNS_Servers:", data[i]['DNS_Servers'])
    print("TACACS_Servers:", data[i]['TACACS_Servers'])
    print("SNMP_Trap_Servers:", data[i]['SNMP_Trap_Servers'])
    print("NTP_Servers:", data[i]['NTP_Servers'])
    print("Logging_Servers:", data[i]['Logging_Servers'])
    print()


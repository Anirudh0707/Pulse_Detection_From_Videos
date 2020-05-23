import numpy as np
import matplotlib.pyplot as plt
import argparse
import json
import heartpy as hp

parser = argparse.ArgumentParser(description='Set the Parameters for Extracting the Waveform from JSON files')
parser.add_argument('-p','--path', type=str, default = None, help='Path to Video File')
parser.add_argument('-f','--frequency', type=int, default = 60, help='Sampling Frequency')
args = parser.parse_args()
if args.path is None:
  print("Add Path Argument")
  print("Format:: python3 file.py --path ./path_to_file")
  exit(1)

with open(args.path) as f:
  data = json.load(f)
print("File :: ", args.path)
key = '/FullPackage'
values = data[key]
data = []
for i in values:
  data.append(i['Value']['waveform'])

# Example of iterable['Value']
# {'barGraph': 5, 'beep': False, 'droppingo2Sat': False, 'o2saturation': 95, 'probeError': False, 'pulseRate': 71, 'searching': False, 'searchingToLong': False, 'signalStrength': 4, 'waveform': 41}
data = np.array(data)
# Display the Data with all essential parameters
working_data, measures = hp.process(data, args.frequency)
hp.plotter(working_data, measures)
print("Average Beats Per Minute :: ", measures['bpm'])
print("Number of Peaks :: ", len(working_data['peaklist']))
# Observed Values
# 0 # 68.11
# 1 # 71.82
# 2 # 53.44
# 3 # 61.18
# 4 # 46.51
# 5 # 65.39
# 6 # 126.89
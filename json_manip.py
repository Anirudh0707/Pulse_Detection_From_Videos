import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import json
import heartpy as hp

ROOT = './Dataset/Data'
JSON_FOLDER = './Dataset/JSON'


with open(os.path.join(JSON_FOLDER,'01-01.json')) as f:
  data = json.load(f)

for i in data.keys():
  print(len(data[i]))
key = '/FullPackage'
values = data[key]
data = []
for i in values:
  data.append(i['Value']['waveform'])

#{'barGraph': 5, 'beep': False, 'droppingo2Sat': False, 'o2saturation': 95, 'probeError': False, 'pulseRate': 71, 'searching': False, 'searchingToLong': False, 'signalStrength': 4, 'waveform': 41}
print(len(data))
data = np.array(data)
plt.plot(data)
plt.show()

working_data, measures = hp.process(data, 60.0)
hp.plotter(working_data, measures)

# 68.11
# 71.82
# 53.44
# 61.18
# 46.51
# 65.39
# 126.89
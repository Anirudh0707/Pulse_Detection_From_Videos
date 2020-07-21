import os
import time

# Automate
ROOT_JSON = './Dataset/JSON/'
ROOT_DATA = './Dataset/Data/'

start_time = []
mid_time   = []
end_time   = []
for dataFile, jsonFile in zip(os.listdir(ROOT_DATA), os.listdir(ROOT_JSON)):
    os.system('python3 json_manip.py --path ' + os.path.join(ROOT_JSON, jsonFile))
    start_time.append(time.time())
    os.system('python3 HeadMovementPulseDet.py --path ' + os.path.join(ROOT_DATA, dataFile))
    mid_time.append(time.time())
    os.system('python3 ColorChangePulseDet.py --path ' + os.path.join(ROOT_DATA, dataFile))
    end_time.append(time.time())

time1 = 0
time2 = 0
for s,m,e in zip(start_time, mid_time, end_time):
    time1 += m - s
    time2 += e - m
    print('Time 1 :: ', m - s)
    print('Time 2 :: ', e - m)
print(time1/len(start_time), time2/len(start_time))


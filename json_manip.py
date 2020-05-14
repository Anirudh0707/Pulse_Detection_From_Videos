import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import json

rootfolder = '.\Problem_Set_Data\Data'
jsonfolder = '.\Problem_Set_Data\JSON'


with open(os.path.join(jsonfolder,'01-01.json')) as f:
  data = json.load(f)

keys = data.keys()
values = data.values()
for i in keys:
    print(type(i))

for i in values:
    print(i[-1])

# # x = os.listdir('./Problem_Set_Data/Data')
# print(os.listdir('.'))

# # cap = cv2.VideoCapture(os.path.join(rootfolder,x[0]))
# cap = cv2.VideoCapture('1.mp4')
# # cap.set(cv2.CV_CAP_PROP_FOURCC, cv2.CV_FOURCC('D','I','V','4'))

# # print(os.path.join(rootfolder,x[0]))    

# if(cap.isOpened() == False):
#     print("Fatal Error")

# while(cap.isOpened()):
#     ret, frame = cap.read()

#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#     cv2.imshow('frame',gray)
#     if cv2.waitKey(10) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()
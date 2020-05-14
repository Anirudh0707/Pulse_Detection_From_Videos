import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import json
import copy

ROOT = './Problem_Set_Data/Data'
JSON_FOLDER = './Problem_Set_Data/JSON'
HAAR = './resources/haarcascade_frontalface_default.xml'


def harrCascadeFaceDet(image):
    faces = face_cascade.detectMultiScale(image, 1.3, 5)
    maxFaceIndex = findMaxFace(faces)

    (x,y,w,h) = faces[maxFaceIndex].copy()
    x = int(x + 0.25 * w)
    w = int(w * 0.5)

    up_y = int(y)
    up_h = int(0.2 * h)
    
    down_y = int(y + 0.55 * h)
    down_h = int(h * 0.45)
    
    upFaceFrame = image[up_y:up_y+up_h, x:x+w]
    downFaceFrame = image[down_y:down_y+down_h, x:x+w]
    
    faceFrame = np.concatenate((upFaceFrame,downFaceFrame), axis=0)
    return faceFrame, faces[maxFaceIndex]
    

def findMaxFace(faces):
    maximum = -1
    for i,(x,y,w,h) in enumerate(faces):
        if(maximum < w * h):
            maximum = w * h
            n = i 
    return n

def interpolationCameraToECG(inputMatrix, inFr, outFr):
    rows, columns = inputMatrix.shape
    Fr = outFr/inFr
    outputMatrix = np.zeros((rows, int(Fr*columns)))
    for i in range(rows):
        inputRow = inputMatrix[i,:]
        inputRow = inputRow.reshape([1,len(inputRow)])
        inputRow = cv2.resize(inputRow, (int(Fr * columns), 1), interpolation = cv2.INTER_CUBIC)
        outputMatrix[i,:] = inputRow
    return outputMatrix
    

if __name__ == '__main__':
    video_filenames = os.listdir(ROOT)
    face_cascade = cv2.CascadeClassifier(HAAR)
    cap = cv2.VideoCapture(os.path.join(ROOT,video_filenames[0]))

    assert cap.isOpened(), 'Cannot capture source'
    ret = True
    while(ret):
        ret, frame = cap.read()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faceFrame, _ = harrCascadeFaceDet(gray)        

        cv2.imshow('frame',faceFrame)
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
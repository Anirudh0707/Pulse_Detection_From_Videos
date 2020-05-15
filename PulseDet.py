import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import json
import copy
from scipy import signal, stats

ROOT = './Dataset/Data'
JSON_FOLDER = './Dataset/JSON'
HAAR = './resources/haarcascade_frontalface_default.xml'

# params for ShiTomasi corner detection
FEATURE_PARAM = dict( maxCorners = 200,
                       qualityLevel = 0.01,
                       minDistance = 5,
                       blockSize = 5 )

# Parameters for lucas kanade optical flow
LK_PARAMS = dict( winSize  = (15,15),
                  maxLevel = 5,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

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

def interpolationAndFiltering(inputMatrix, inFr, outFr, lowerCutoff=0.75, higherCutoff=5, filterOrder=3):
    rows, columns = inputMatrix.shape
    Fr = outFr/inFr
    outputMatrix = np.zeros((int(Fr*rows), columns))
    for i in range(rows):
        #Interpolate the Data
        inputCol = inputMatrix[:,i]
        inputCol = cv2.resize(inputCol.reshape([len(inputCol),1]), (1, int(Fr * rows)), interpolation = cv2.INTER_CUBIC)

        # Transfrom the cutoff frequencies from the analog domain to the digital domain
        lowerCutoffDigital = lowerCutoff / (0.5 * outFr)
        higherCutoffDigital = higherCutoff / (0.5 * outFr)
        # Filter the data with a Butterworth bandpass filter and a filtfilt operation for a zero-phase response
        b, a = signal.butter(filterOrder, [lowerCutoffDigital, higherCutoffDigital], btype='band')
        outputCol = signal.filtfilt(b, a, inputCol.ravel())
        outputMatrix[:,i] = outputCol.ravel()
    return outputMatrix
    
def processRawData(rawData, status, threshold = 0.25):
    rawData = rawData[:,:,1]
    rawData = rawData[:, status.ravel() == 1]
    return rawData



if __name__ == '__main__':
    video_filenames = os.listdir(ROOT)
    face_cascade = cv2.CascadeClassifier(HAAR)
    cap = cv2.VideoCapture(os.path.join(ROOT,video_filenames[0]))
    assert cap.isOpened(), 'Cannot capture source'

    # First Frame processing
    ret = True
    counter = 0
    cornerList = []
    while(ret):
        ret, frame = cap.read()
        newFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faceFrame, _ = harrCascadeFaceDet(newFrame)
        # Find Corner and then Track it
        if counter == 0:
            # Remember the First Frame and its Size
            savedFrame = faceFrame
            savedSize  = faceFrame.shape[::-1]
            # Extract the Shi-Tomasi Corners
            p0 = cv2.goodFeaturesToTrack(savedFrame, mask = None, **FEATURE_PARAM)
            cornerList.append(p0.reshape(-1,2))
            status = np.ones((len(p0),1))
        else :
            faceFrame = cv2.resize(faceFrame, savedSize)
            # Use the Lucas Kanade algorithm to determine the feature point locations in the new image
            p1, st, err = cv2.calcOpticalFlowPyrLK(savedFrame, faceFrame, p0, None, **LK_PARAMS)
            cornerList.append(p1.reshape(-1,2))
            # Will be used to remove uncertain points
            status *= st
        # Draw Markers on the Feature Locations
        for i in cornerList[counter]:
            x,y = i.ravel()
            cv2.circle(faceFrame, (x,y), 1, 255, -1)
        cv2.imshow('frame',faceFrame)
        counter += 1
        if cv2.waitKey(3) & 0xFF == ord('q'):
            break
    # Release the Video capture and frame
    cap.release()
    cv2.destroyAllWindows()
    print("End of Feature Tracking")
    print("Counter ", counter)
    print("Number of Features worth retaining ", int(np.sum(status)) )

    # Create the Corner feature Matrix
    rawData = np.array(cornerList)
    # Process the Data toi remove extremities
    processedData = processRawData(rawData,status)
    # Interpolate and filter the processed data
    filteredData = interpolationAndFiltering(processedData,30,60)
    print(filteredData.shape)    
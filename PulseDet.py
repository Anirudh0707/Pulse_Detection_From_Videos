import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import json
import copy
import heartpy as hp
from scipy import signal, stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

ROOT = './Dataset/Data'
JSON_FOLDER = './Dataset/JSON'
HAAR = './resources/haarcascade_frontalface_default.xml'

# params for ShiTomasi corner detection
FEATURE_PARAM = dict( maxCorners = 200,
                       qualityLevel = 0.001,
                       minDistance = 7,
                       blockSize = 7 )

# Corner Sub pixel
CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)

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
    h = int(h * 0.9)

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

def interpolationAndFiltering(inputMatrix, inFr, outFr, lowerCutoff=0.75, higherCutoff=5, filterOrder=5, savgolWindow = 67, polynomialOrder=5):
    rows, columns = inputMatrix.shape
    Fr = outFr/inFr
    outputMatrix = np.zeros((int(Fr*rows), columns))
    for i in range(columns):
        #Interpolate the Data
        inputCol = inputMatrix[:,i]
        inputCol = cv2.resize(inputCol.reshape([len(inputCol.ravel()),1]), (1, int(Fr * rows)), interpolation = cv2.INTER_CUBIC)
        # Transfrom the cutoff frequencies from the analog domain to the digital domain
        lowerCutoffDigital = lowerCutoff / (0.5 * outFr)
        higherCutoffDigital = higherCutoff / (0.5 * outFr)
        # Filter the data with a Butterworth bandpass filter and a filtfilt operation for a zero-phase response
        b, a = signal.butter(filterOrder, [lowerCutoffDigital, higherCutoffDigital], btype='band', analog=False)
        outputCol = signal.filtfilt(b, a, inputCol.ravel())
        outputCol = signal.savgol_filter(outputCol, savgolWindow, polynomialOrder)
        outputMatrix[:,i] = outputCol.ravel()
    return outputMatrix
    
def processRawData(rawData, status, threshold = 0.25):
    # Remove Feature that couldnt be tracked
    rawData = np.round(rawData[:,:,1])
    tempData = np.abs(rawData[1:, :] - rawData[0:-1, :])
    # Remove unstable features
    maxData  = np.amax(tempData, axis = 0).ravel()
    modeData, _ = stats.mode(maxData, axis = None)
    processedData = rawData[:, maxData <= modeData]
    print("Stable Data Shape ", processedData.shape)
    return processedData

def computePCA(filteredData, n_components = 5, alpha = 0.25):
    tempData = filteredData.copy()
    # Remove top 25% using the L-2 norm
    normData = np.linalg.norm(tempData, axis = 1).ravel()
    indicies = np.argsort(normData)
    topLimit = int( (1 - alpha) * len(indicies))
    # Sorted in ascending order. Hence, take time instances lesser than the top 25 %
    tempData = tempData[np.sort(indicies[:topLimit]).ravel(), :] 
    # Fit the PCA model
    scalar = StandardScaler()
    standardizedData = scalar.fit_transform(tempData)
    pca = PCA(n_components = n_components)
    pca.fit(standardizedData)
    # Apply the PCA model
    filteredData = scalar.transform(filteredData)
    principalComponents = pca.transform(filteredData)
    return principalComponents


if __name__ == '__main__':
    video_filenames = os.listdir(ROOT)
    face_cascade = cv2.CascadeClassifier(HAAR)
    cap = cv2.VideoCapture(os.path.join(ROOT,video_filenames[6]))
    print(os.path.join(ROOT,video_filenames[0]))
    assert cap.isOpened(), 'Cannot capture source'

    # First Frame processing
    counter = 0
    cornerList = []
    while(True):
        ret, frame = cap.read()
        if ret == False:
            break
        newFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faceFrame, _ = harrCascadeFaceDet(newFrame)
        # Find Corner and then Track it
        if counter == 0:
            # Remember the First Frame and its Size
            savedFrame = faceFrame
            savedSize  = faceFrame.shape[::-1]
            # Extract the Shi-Tomasi Corners
            p0 = cv2.goodFeaturesToTrack(savedFrame, mask = None, **FEATURE_PARAM)
            p0 = cv2.cornerSubPix(savedFrame,p0,(15,15),(-1,-1),CRITERIA)
            cornerList.append(p0.reshape(-1,2))
        else :
            faceFrame = cv2.resize(faceFrame, savedSize)
            # Use the Lucas Kanade algorithm to determine the feature point locations in the new image
            p1, st, err = cv2.calcOpticalFlowPyrLK(savedFrame, faceFrame, p0, None, **LK_PARAMS)
            cornerList.append(p1.reshape(-1,2))
        # Draw Markers on the Feature Locations
        for i in cornerList[counter]:
            x,y = i.ravel()
            cv2.circle(faceFrame, (x,y), 1, 255, -1)
        cv2.imshow('frame',faceFrame)
        counter += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # Release the Video capture and frame
    cap.release()
    cv2.destroyAllWindows()
    print("Counter ", counter)

    # Create the Corner feature Matrix
    rawData = np.array(cornerList)
    # Process the Data toi remove extremities
    processedData = processRawData(rawData,status=None)
    # Interpolate and filter the processed data
    filteredData = interpolationAndFiltering(processedData,30,60)
    print("Filtered and Interpolated Data Shape ", filteredData.shape)
    # PCA decomposition and projecting onto the best vector
    principalComponents = computePCA(filteredData)
    nyquist = int(len(principalComponents)/2)
    powerRatio = []
    for i in range(principalComponents.shape[1]):
        fftData = np.fft.fft(principalComponents[:,i])[1:nyquist]
        powerSpectrum = np.abs(fftData)**2
        maxFreq = np.argmax(powerSpectrum)
        print(maxFreq/nyquist*30)
        powerInMaxFreq = np.sum(powerSpectrum[maxFreq-1:maxFreq+2]) + np.sum(powerSpectrum[2*maxFreq:2*maxFreq+3])
        powerRatio.append(powerInMaxFreq/np.sum(powerSpectrum))
    print(powerRatio)
    # Plot
    x_disp = 30*np.arange(nyquist)/nyquist
    x_disp = x_disp[1:]

    for i in range(principalComponents.shape[1]):
        y_disp = np.fft.fft(principalComponents[:,i])
        plt.plot(x_disp, np.abs(y_disp[1:nyquist]))
        plt.show()
    
    for i in range(principalComponents.shape[1]):
        y_disp = principalComponents[:,i]
        # working_data, measures = hp.process(y_disp, 30.0)
        # hp.plotter(working_data, measures)
        plt.plot(y_disp)
        plt.show()
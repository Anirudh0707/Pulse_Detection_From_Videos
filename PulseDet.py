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
FEATURE_PARAM = dict( maxCorners = 50,
                       qualityLevel = 0.01,
                       minDistance = 7,
                       blockSize = 7 )

# Corner Sub pixel
CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)

# Parameters for lucas kanade optical flow
LK_PARAMS = dict( winSize  = (10,10),
                  maxLevel = 5,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

def harrCascadeFaceDet(image):
    faces = face_cascade.detectMultiScale(image, 1.3, 5)
    maxFaceIndex = findMaxFace(faces)
    (x,y,w,h) = faces[maxFaceIndex].copy()
    x = int(x + 0.25 * w)
    w = int(w * 0.5)
    h = int(h * 0.9)
    faceFrame = getFace(image, (x,y,w,h))
    return faceFrame, (x,y,w,h)

def getFace(image, faceTuple):
    (x,y,w,h) = faceTuple
    faceFrame = image[y:y + h, x:x+w]
    faceFrame[int(0.2*h):int(0.55*h), :] = 0
    return faceFrame
    

def findMaxFace(faces):
    maximum = -1
    for i,(x,y,w,h) in enumerate(faces):
        if(maximum < w * h):
            maximum = w * h
            n = i 
    return n

def interpolationAndFiltering(inputMatrix, inFr, outFr, lowerCutoff=0.75, higherCutoff=4, filterOrder=5, savgolWindow = 61, polynomialOrder=7):
    rows, columns = inputMatrix.shape
    # Transfrom the cutoff frequencies from the analog domain to the digital domain
    lowerCutoffDigital = lowerCutoff / (0.5 * outFr)
    higherCutoffDigital = higherCutoff / (0.5 * outFr)
    Fr = outFr/inFr
    outputMatrix = np.zeros((int(Fr*rows), columns))
    for i in range(columns):
        #Interpolate the Data
        inputCol = inputMatrix[:,i]
        # # Filter the data with a Butterworth bandpass filter and a filtfilt operation for a zero-phase response
        # b, a = signal.butter(filterOrder, [lowerCutoffDigital*Fr, higherCutoffDigital*Fr], btype='band', analog=False)
        # inputCol = signal.filtfilt(b, a, inputCol.ravel())
        
        # Interpolate
        inputCol = cv2.resize(inputCol.reshape([len(inputCol.ravel()),1]), (1, int(Fr * rows)), interpolation = cv2.INTER_CUBIC)
        # Filter the data with a Butterworth bandpass filter and a filtfilt operation for a zero-phase response
        b, a = signal.butter(filterOrder, [lowerCutoffDigital, higherCutoffDigital], btype='band', analog=False)
        inputCol = signal.filtfilt(b, a, inputCol.ravel())
        # inputCol = signal.savgol_filter(inputCol, savgolWindow, polynomialOrder)
        outputMatrix[:,i] = inputCol.ravel()
    return outputMatrix
    
def processRawData(rawData, status):
    # Remove Feature that couldnt be tracked
    rawData = rawData[:,:,1]
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
    meanRow = np.mean(tempData, axis = 0)
    tempData =  tempData - meanRow

    CovMat = np.transpose(tempData) @ (tempData)
    eigenValues, eigenVectors = np.linalg.eig(CovMat)
    indiciesEigen = np.argsort(np.real(eigenValues))
    indiciesEigen = indiciesEigen[::-1]
    indiciesEigen = indiciesEigen[0:n_components]
    eigenVectors = np.real(eigenVectors[:,indiciesEigen])    
    principalComponents = filteredData @ eigenVectors
    ###############################
    # pca = PCA(n_components = n_components)
    # pca.fit(tempData)
    # # Apply the PCA model
    # principalComponents = pca.transform(filteredData)
    return principalComponents


if __name__ == '__main__':
    video_filenames = os.listdir(ROOT)
    face_cascade = cv2.CascadeClassifier(HAAR)
    cap = cv2.VideoCapture(os.path.join(ROOT,video_filenames[6]))
    assert cap.isOpened(), 'Cannot capture source'

    # First Frame processing
    counter = 0
    inFr  = 30
    outFr = 60
    cornerList = []
    while(True):
        ret, frame = cap.read()
        if ret == False:
            break
        newFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Find Corner and then Track it
        if counter == 0:
            faceFrame, faceTuple = harrCascadeFaceDet(newFrame)
            # Remember the First Frame and its Size
            savedFrame = faceFrame
            # Extract the Shi-Tomasi Corners
            p0 = cv2.goodFeaturesToTrack(savedFrame, mask = None, **FEATURE_PARAM)
            p0 = cv2.cornerSubPix(savedFrame,p0,(15,15),(-1,-1),CRITERIA)
            cornerList.append(np.round(p0.reshape(-1,2)))
        else :
            faceFrame = getFace(newFrame, faceTuple)
            # Use the Lucas Kanade algorithm to determine the feature point locations in the new image
            p1, st, err = cv2.calcOpticalFlowPyrLK(savedFrame, faceFrame, p0, None, **LK_PARAMS)
            cornerList.append(np.round(p1.reshape(-1,2)))
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
    cornerList.pop(0)
    rawData = np.array(cornerList)
    # Process the Data toi remove extremities
    processedData = processRawData(rawData,status=None)
    # Interpolate and filter the processed data
    filteredData = interpolationAndFiltering(processedData,inFr,outFr)
    print("Filtered and Interpolated Data Shape ", filteredData.shape)
    # PCA decomposition and projecting onto the best vector
    principalComponents = computePCA(filteredData)
    nyquist = int(len(principalComponents)/2)
    powerRatio = []
    for i in range(principalComponents.shape[1]):
        fftData = np.fft.fft(principalComponents[:,i])[1:nyquist]
        powerSpectrum = np.abs(fftData)**2
        maxFreq = np.argmax(powerSpectrum)
        print(i, "     ", (maxFreq+1)/nyquist*outFr/2)
        powerInMaxFreq = np.sum(powerSpectrum[maxFreq-1:maxFreq+2]) #+ np.sum(powerSpectrum[2*maxFreq:2*maxFreq+3])
        powerRatio.append(powerInMaxFreq/np.sum(powerSpectrum))
    print(powerRatio)
    PCAIndex = np.argmax(np.array(powerRatio))
    chosenSignal = principalComponents[:,PCAIndex]
    print(PCAIndex)
    # Plot
    x_disp = outFr/2*np.arange(nyquist)/nyquist
    x_disp = x_disp[1:]

    for i in range(principalComponents.shape[1]):
        y_disp = np.fft.fft(principalComponents[:,i])
        plt.plot(x_disp, np.abs(y_disp[1:nyquist]))
        plt.title(str(i))
        plt.show()
    
    for i in range(principalComponents.shape[1]):
        y_disp = principalComponents[:,i]
        # working_data, measures = hp.process(y_disp, 60.0)
        # hp.plotter(working_data, measures)
        plt.plot(y_disp)
        plt.title(str(i))
        plt.show()

    # Peak det
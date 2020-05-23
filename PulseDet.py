import numpy as np
import matplotlib.pyplot as plt
import argparse
import cv2
from scipy import signal, stats
from sklearn.decomposition import PCA, FastICA

GT = (68.11, 71.82, 53.44, 61.18, 46.51, 65.39, 126.89)

# params for ShiTomasi corner detection
FEATURE_PARAM = dict( maxCorners = 100,
                       qualityLevel = 0.01,
                       minDistance = 7,
                       blockSize = 7 )

# Corner Sub pixel
CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)

# Parameters for lucas kanade optical flow
LK_PARAMS = dict( winSize  = (100,5),
                  maxLevel = 17,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 40, 0.01))

def parseArguments():
    parser = argparse.ArgumentParser(description='Set the Parameters for Video Processing')
    parser.add_argument('-p','--path', type=str, default = 'face.mp4', help='Path to Video File')
    parser.add_argument('-o','--outputSamplingFrequency', type=int, default = 60, help='Number of levels for the image pyramid')
    parser.add_argument('-n','--numComponents', type=int, default = 5, help='Number of components for PCA')
    parser.add_argument('-a','--alpha', type=int, default = 25, help='Int between 0 to 100. Top alpha percent is discarded')
    parser.add_argument('-q','--qfactor', type=int, default = 2, help='Q Factor for FFT peak enhancement')

    args = parser.parse_args()
    return args.path, args.outputSamplingFrequency, args.numComponents , args.alpha, args.qfactor

def harrCascadeFaceDet(image):
    face_cascade = cv2.CascadeClassifier('./resources/haarcascade_frontalface_default.xml')
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

def interpolationAndFiltering(inputMatrix, inFr, outFr, lowerCutoff=0.75, higherCutoff=4, filterOrder=5):
    rows, columns = inputMatrix.shape
    # Transfrom the cutoff frequencies from the analog domain to the digital domain
    lowerCutoffDigital = lowerCutoff / (0.5 * outFr)
    higherCutoffDigital = higherCutoff / (0.5 * outFr)
    Fr = outFr/inFr
    outputMatrix = np.zeros((int(Fr*rows), columns))
    for i in range(columns):
        #Interpolate the Data
        inputCol = inputMatrix[:,i]
        inputCol = cv2.resize(inputCol.reshape([len(inputCol.ravel()),1]), (1, int(Fr * rows)), interpolation = cv2.INTER_CUBIC)
        # Filter the data with a Butterworth bandpass filter and a filtfilt operation for a zero-phase response
        b, a = signal.butter(filterOrder, [lowerCutoffDigital, higherCutoffDigital], btype='band', analog=False)
        inputCol = signal.filtfilt(b, a, inputCol.ravel())
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
    pca = PCA(n_components = n_components)
    pca.fit(tempData)
    # Apply the PCA model
    principalComponents = pca.transform(filteredData)
    return principalComponents

def peakAmplification(chosenSignal, outFr, f0, Q):
    b1, a1 = signal.iirpeak(f0, Q, outFr)
    peakFiltered = signal.filtfilt(b1, a1, chosenSignal.ravel())
    b2, a2 = signal.iirpeak(2*f0, Q, outFr)
    harmonicFiltered = signal.filtfilt(b2, a2, chosenSignal.ravel())
    return peakFiltered + harmonicFiltered

if __name__ == '__main__':
    path, outFr, n_components, removeTopComponents, Q_Factor = parseArguments()
    cap = cv2.VideoCapture(path)
    assert cap.isOpened(), 'Cannot capture source'
    inFr  = int(cap.get(cv2.CAP_PROP_FPS))

    counter = 0
    cornerList = []
    while(True):
        ret, frame = cap.read()
        if ret == False:
            break
        newFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Find Corner and then Track it
        if counter == 0:
            faceFrame, faceTuple = harrCascadeFaceDet(newFrame)
            # Remember the First Frame
            savedFrame = faceFrame
            # Extract the  Corners
            p0 = cv2.goodFeaturesToTrack(savedFrame, mask = None, **FEATURE_PARAM)
            p0 = cv2.cornerSubPix(savedFrame,p0,(15,15),(-1,-1),CRITERIA)
            cornerList.append(np.round(p0.reshape(-1,2)))
        else :
            faceFrame = getFace(newFrame, faceTuple)
            # Use the Lucas Kanade algorithm to determine the feature point locations in the new image
            p1, st, err = cv2.calcOpticalFlowPyrLK(savedFrame, faceFrame, p0, None, **LK_PARAMS)
            cornerList.append(np.round(p1.reshape(-1,2)))
        # Please Do Not  Alter This Loop. Altering This Loop Is Currently Leading To Data Corruption
        for i in cornerList[counter]:
            x,y = i.ravel()
            cv2.circle(faceFrame, (x,y), 1, 255, -1)
        counter += 1
    # Release the Video capture and frame
    cap.release()
    # Create the Corner feature Matrix
    cornerList.pop(0)
    rawData = np.array(cornerList)
    # Process the Data to remove extremities
    processedData = processRawData(rawData,status=None)
    # Interpolate and filter the processed data
    filteredData = interpolationAndFiltering(processedData,inFr,outFr)
    print("Filtered and Interpolated Data Shape ", filteredData.shape)
    # PCA decomposition and projecting onto the best vector
    principalComponents = computePCA(filteredData, n_components = n_components, alpha = removeTopComponents/100)
    nyquist = int(len(principalComponents)/2)
    powerRatio = []
    listForDistanceEstimation = []
    for i in range(principalComponents.shape[1]):
        fftData = np.fft.fft(principalComponents[:,i])[1:nyquist]
        powerSpectrum = np.abs(fftData)**2
        maxFreq = np.argmax(powerSpectrum)
        # print(i, "     ", (maxFreq+1)/nyquist*outFr/2)
        powerInMaxFreq = np.sum(powerSpectrum[maxFreq-1:maxFreq+2]) #+ np.sum(powerSpectrum[2*maxFreq:2*maxFreq+3])
        powerRatio.append(powerInMaxFreq/np.sum(powerSpectrum))
        listForDistanceEstimation.append((maxFreq+1)/nyquist*outFr/2)
    PCAIndex = np.argmax(np.array(powerRatio))
    chosenSignal = principalComponents[:,PCAIndex]
    chosenSignal = peakAmplification(chosenSignal, outFr = outFr, f0 = listForDistanceEstimation[PCAIndex], Q = Q_Factor)
    # Plot
    x_disp = outFr/2*np.arange(nyquist)/nyquist
    x_disp = x_disp[1:]
    distance = int(outFr/listForDistanceEstimation[PCAIndex])-5 # Emperically found realtion
    print("Peak Min Dist ", distance)
    peaks, _ = signal.find_peaks(chosenSignal, distance=distance)
    plt.plot(chosenSignal)
    plt.plot(peaks, chosenSignal[peaks], "x")
    plt.title("Avg Heart Beat = "+str(listForDistanceEstimation[PCAIndex]*60))
    plt.show()
    print("Average Beats Per Minute :: " + str(listForDistanceEstimation[PCAIndex]*60))
    print("Number of Peaks :: ", len(peaks))
    # y_disp = np.fft.fft(chosenSignal)
    # plt.plot(x_disp, np.abs(y_disp[1:nyquist]))
    # plt.show()
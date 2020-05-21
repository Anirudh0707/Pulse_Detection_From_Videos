import numpy as np
import argparse
import matplotlib.pyplot as plt
import cv2
import EVM
from scipy import signal, stats
from sklearn.decomposition import PCA, FastICA

GT = (68.11, 71.82, 53.44, 61.18, 46.51, 65.39, 126.89)

def parseArguments():
    parser = argparse.ArgumentParser(description='Set the Parameters for Video Processing')
    parser.add_argument('-p','--path', type=str, default = 'face.mp4', help='Path to Video File')
    parser.add_argument('-l','--levels', type=int, default = 3, help='Number of Levels for the Image Pyramid')
    parser.add_argument('-a','--amplification', type=int, default = 20, help='Amplification Factor for Video Magnification')
    parser.add_argument('-o','--outputSamplingFrequency', type=int, default = 60, help='Number of Levels for the Image Pyramid')

    args = parser.parse_args()
    return args.path, args.levels, args.amplification, args.outputSamplingFrequency

def harrCascadeFaceDet(image):
    face_cascade = cv2.CascadeClassifier('./resources/haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(image, 1.3, 5)
    maxFaceIndex = findMaxFace(faces)
    (x,y,w,h) = faces[maxFaceIndex].copy()
    x = int(x + 0.25 * w)
    w = int(w * 0.5)
    h = int(h * 0.9)
    w = w + int(2**levels)
    w = w - (w % int(2**levels))
    h = h + int(2**levels)
    h = h - (h % int(2**levels))
    return (x,y,w,h)

def getFace(image, faceTuple):
    (x,y,w,h) = faceTuple
    faceFrame = image[y:y + h, x:x+w, :]
    faceFrame[int(0.2*h):int(0.55*h), :, :] = 0
    return faceFrame
    

def findMaxFace(faces):
    maximum = -1
    for i,(x,y,w,h) in enumerate(faces):
        if(maximum < w * h):
            maximum = w * h
            n = i 
    return n

def interpolationAndFiltering(inputMatrix, inFr, outFr, lowerCutoff=0.75, higherCutoff=5, filterOrder=5, savgolWindow = 61, polynomialOrder=6):
    rows, columns = inputMatrix.shape
    # Transfrom the cutoff frequencies from the analog domain to the digital domain
    lowerCutoffDigital = lowerCutoff / (0.5 * outFr)
    higherCutoffDigital = higherCutoff / (0.5 * outFr)
    Fr = outFr/inFr
    outputMatrix = np.zeros((int(Fr*rows), columns))
    for i in range(columns):
        #Interpolate the Data
        inputCol = inputMatrix[:,i]
        # Interpolate
        inputCol = cv2.resize(inputCol.reshape([len(inputCol.ravel()),1]), (1, int(Fr * rows)), interpolation = cv2.INTER_CUBIC)
        # Filter the data with a Butterworth bandpass filter and a filtfilt operation for a zero-phase response
        b, a = signal.butter(filterOrder, [lowerCutoffDigital, higherCutoffDigital], btype='band', analog=False)
        inputCol = signal.filtfilt(b, a, inputCol.ravel())
        # inputCol = signal.savgol_filter(inputCol, savgolWindow, polynomialOrder)
        outputMatrix[:,i] = inputCol.ravel()
    return outputMatrix

if __name__ == '__main__':
    path, levels, amplification, outFr = parseArguments()
    cap = cv2.VideoCapture(path) 
    assert cap.isOpened(), 'Cannot capture source'
    inFr  = int(cap.get(cv2.CAP_PROP_FPS))

    counter = 0
    videoTensor = []
    while(True):
        ret, frame = cap.read()
        if ret == False:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if counter == 0:
            faceTuple = harrCascadeFaceDet(gray)
        videoTensor.append(getFace(frame,faceTuple))        
        counter += 1
    cap.release()
    videoTensor = np.array(videoTensor)
    print("Video Tensor Ready")
    gau_video = EVM.gaussian_video(videoTensor,levels=levels)
    print("Pyramid Fomation Complete")
    filtered_tensor = EVM.temporal_ideal_filter(gau_video,0.8,3,inFr)
    print("Filter Complete")
    amplifiedVideo = filtered_tensor * amplification
    reconstructedTensor = EVM.reconstructVideo(amplifiedVideo,videoTensor,levels=levels)
    print("Reconstruction Complete")

    # Sum over Width and Breadth to get the Time vs RGB components
    # Note : Format is B,G,R due to OpenCV
    BGRComponents = np.mean(reconstructedTensor, axis = 1)
    BGRComponents = np.mean(BGRComponents, axis = 1)
    BGRComponents = interpolationAndFiltering(BGRComponents, inFr, outFr)
    # Extract Components
    model = PCA(n_components = 3)
    principalComponents = model.fit_transform(BGRComponents)
    nyquist = int(len(principalComponents)/2)  
    x_disp = outFr/2*np.arange(nyquist)/nyquist
    # Selecte Best Component. Usually the Second one i.e Index = 1
    powerRatio = []
    listForDistanceEstimation = []
    for i in range(3):
        fftData = np.fft.fft(principalComponents[:,i])[0:nyquist]
        powerSpectrum = np.abs(fftData)**2
        maxFreq = np.argmax(powerSpectrum)
        powerInMaxFreq = np.sum(powerSpectrum[maxFreq-1:maxFreq+2]) + np.sum(powerSpectrum[2*maxFreq:2*maxFreq+3])
        powerRatio.append(powerInMaxFreq/np.sum(powerSpectrum))
        listForDistanceEstimation.append((maxFreq)/nyquist*outFr/2)
    # Choose Signal 
    PCAIndex = np.argmax(np.array(powerRatio))
    chosenSignal = principalComponents[:,PCAIndex]
    # Peak Det and Display
    distance = int(outFr*2/listForDistanceEstimation[PCAIndex])-10 # Emperically found realtion
    print(distance)
    peaks, _ = signal.find_peaks(chosenSignal, distance=distance)
    plt.plot(chosenSignal)
    plt.plot(peaks, chosenSignal[peaks], "x")
    plt.title("Avg Heart Beat = "+str(60*60/(peaks[-1] - peaks[0])*len(peaks)))
    plt.show()

    y_disp = np.fft.fft(chosenSignal)
    plt.plot(x_disp, np.abs(y_disp[0:nyquist]))
    plt.show()
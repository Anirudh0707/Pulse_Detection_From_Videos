import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import json
import copy
import heartpy as hp
from scipy import signal, stats
from sklearn.decomposition import PCA, FastICA
from sklearn.preprocessing import StandardScaler
import EVM

ROOT = './Dataset/Data'
JSON_FOLDER = './Dataset/JSON'
HAAR = './resources/haarcascade_frontalface_default.xml'
GT = (68.11, 71.82, 53.44, 61.18, 46.51, 65.39, 126.89)
LEVELS = 3
AMPLIFICATION = 20



def harrCascadeFaceDet(image):
    faces = face_cascade.detectMultiScale(image, 1.3, 5)
    maxFaceIndex = findMaxFace(faces)
    (x,y,w,h) = faces[maxFaceIndex].copy()
    x = int(x + 0.25 * w)
    w = int(w * 0.5)
    h = int(h * 0.9)
    w = w + int(2**LEVELS)
    w = w - (w % int(2**LEVELS))
    h = h + int(2**LEVELS)
    h = h - (h % int(2**LEVELS))
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
    video_filenames = os.listdir(ROOT)
    face_cascade = cv2.CascadeClassifier(HAAR)
    inputNumer = int(input("Enter a Num from 0 to 6 :: "))
    cap = cv2.VideoCapture(os.path.join(ROOT,video_filenames[inputNumer]))
    assert cap.isOpened(), 'Cannot capture source'

    counter = 0
    inFr  = 30
    outFr = 60
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    assert fps == inFr
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
    print(videoTensor.shape)
    print("Video Tensor Ready")
    gau_video = EVM.gaussian_video(videoTensor,levels=LEVELS)
    print("Pyramid Fomation Complete")
    filtered_tensor = EVM.temporal_ideal_filter(gau_video,0.8,3,inFr)
    print("Filter Complete")
    amplifiedVideo = filtered_tensor * AMPLIFICATION
    reconstructedTensor = EVM.reconstructVideo(amplifiedVideo,videoTensor,levels=LEVELS)
    print("Reconstruction Complete")
    # EVM.save_video(reconstructedTensor)
    print(reconstructedTensor.shape)

    # Sum over Width and Breadth to get the Time vs RGB components
    # Note 1: Format is BGR due to OpenCV
    # Note 2: Using 2 lines for computing mean as only the recent numpy library has multi axis summing in a sinlge command(acc to th numpy doc website)
    BGRComponents = np.mean(reconstructedTensor, axis = 1)
    BGRComponents = np.mean(BGRComponents, axis = 1)
    # BGRComponents = np.mean(reconstructedTensor, axis = (1,2)) # Use when numpy>=1.7.0
    print(BGRComponents.shape)
    # PCA 
    pca = PCA(n_components = 3)
    principalComponents = pca.fit_transform(BGRComponents)
    # for i in range(3):
    #     fftData = np.fft.fft()
    # Or Just take Green Channel
    outputSignal = BGRComponents[:,1]
    plt.plot(outputSignal)
    plt.show()
    

# Pulse Detection From Videos

A project aimed at detecting the human pulse rate from video recordings. This can be used for non-contact monitoring of a patient's health.

We have tackled this problems from 2 perspectives, namely the head movements tracking and the facial color tracking algotihms.

## Libraries Used

numpy, scipy, opencv, matplotlib, sklearn, heartpy, argparse, json


## Pulse Detection From Movements of the Head

To run the algorithm, open the command prompt and enter the folowing line
```
python3 HeadMovementPulseDet.py --path FilePath
```
There are other arguments that can be parsed. For more information, please run
```
python3 HeadMovementPulseDet.py --help
```

This code takes in an input video file from the dataset and runs the algorithm descirbed in this [paper](https://people.csail.mit.edu/mrub/vidmag/papers/Balakrishnan_Detecting_Pulse_from_2013_CVPR_paper.pdf). The algorithms finds the Shi-Tomasi corners for the frist frame and tracks their movements in the subsequent frames using the Lucas Kanade optic flow algorithm. The y displacement are taken and post processed to obtain the best signals. These are then interpolated filtered and decomposed using the PCA. The final signal is chosen amongs the decomposed signals and is peak amplified

An additional Jupyter notebook has been added to delinate each aspect of the code.

## Pusle Detection From Color Changes in the Head

We have used a plug and play Video Magnification Code for amplifying the color changes in the facial regions.<br/>
Please note that the orginal repo can be found at [here](https://github.com/flyingzhao/PyEVM)

To run the algorithm, open the command prompt and enter the folowing line
```
python3 ColorChangePulseDet.py --path FilePath
```
There are other arguments that can be parsed. For more information, please run
```
python3 ColorChangePulseDet.py --help
```

This code takes in an input video file from the dataset and runs the algorithm descirbed in this [paper](https://people.csail.mit.edu/mrub/papers/vidmag.pdf) to first amplify the color of the video. The algorithm then takes the color amplified signal and spatially averages over the entire height and width of each time instance and color channel. The 3 signals are then interpolated and filtered. The 3 signals are decomposed using the PCA. The final signal is chosen among the 3 decomposed signals and is peak amplififed 

An additional Jupyter notebook has been added to delinate each aspect of the code.

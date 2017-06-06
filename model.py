import csv
import cv2 
import numpy as np
# Data read from csv. CSV contains the image path and steerig angles recorded at various points of gameplay.
lines = []

csvPath = "/Users/hackintoshrao/Documents/code/self-drive/driving_log.csv"

with open(csvPath) as csvData:
	# read from csv 
	reader = csv.reader(csvData)
	for line in reader:
		lines.append(line)

# images from gameplay are read from disk and queued here.
cameraImages = []
# corresponding steering measurements are queued up here.
steeringMeasurements = []

for line in lines:
	imgPath = line[0]
	img = cv2.imread(imgPath)
	cameraImages.append(image)

	measurement = float(line[3])
	steeringMeasurements.append(measurements)

# convert the images and measurements to numpy array.
xTrain = np.array(cameraImages)
yTrain = np.array(steeringMeasurements)



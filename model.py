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
	cameraImages.append(img)

	measurement = float(line[3])
	steeringMeasurements.append(measurement)

# convert the images and measurements to numpy array.
xTrain = np.array(cameraImages)
yTrain = np.array(steeringMeasurements)

# Using keras to build and train the model.
from keras.models import Sequential
from keras.layers import Flatten, Dense 
# building a sequential regression network (not an classification network).
# the model will be trained on the image and using streeing measurements as output data.
# single output node will predict the streeing measurement.
model = Sequential()
# the input image is of dimention 160 x 320 x 3
model.add(Flatten(input_shape=(160,320,3)))
# the network has just one output node.
model.add(Dense(1))
# Defining mean square error to be the loss functions and using Adam optimizer.
model.compile(loss='mse', optimizer='adam')

model.fit(xTrain, yTrain, validation_split=0.2, shuffle=True)

model.save('model.h5')


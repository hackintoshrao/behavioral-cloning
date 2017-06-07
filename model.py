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
	# Flipping the image so the network is trained to move in the mirror imaged path too.
	# This also help generate more training data.
	flippedImg = np.fliplr(img)
	# add the original image and the flipped image to the list of camera images.
	cameraImages.append(img)
	cameraImages.append(flippedImg)

	measurement = float(line[3])
	# measurement of steering angle corresponding to the flipped img.
	measurementFlipped = -measurement
	# append the steering measurement for the original image and the flipped image.
	steeringMeasurements.append(measurement)
	steeringMeasurements.append(measurementFlipped)

# convert the images and measurements to numpy array.
xTrain = np.array(cameraImages)
yTrain = np.array(steeringMeasurements)

# Using keras to build and train the model.
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda
# building a sequential regression network (not an classification network).
# the model will be trained on the image and using streeing measurements as output data.
# single output node will predict the streeing measurement.
model = Sequential()
# preprocess data.
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160, 320, 3)))

# the input image is of dimention 160 x 320 x 3
model.add(Flatten())
# the network has just one output node.
model.add(Dense(1))
# Defining mean square error to be the loss functions and using Adam optimizer.
model.compile(loss='mse', optimizer='adam')
# Train the model.
# 20% of data is reserved for validation data.
model.fit(xTrain, yTrain, validation_split=0.2, shuffle=True, nb_epoch=4)
# save the trained model.
model.save('model.h5')


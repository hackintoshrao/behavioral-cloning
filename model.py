import csv
import cv2
import numpy as np
import random

# Using keras to build and train the model.
from keras.models import Sequential
from keras.layers import Cropping2D
from keras.layers import Input, Flatten, Dense, Lambda, Cropping2D, Dropout, ELU
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D


def process_get_data(line):
	imgPath = line[0]
	steering = float(line[3])
	# randomly choose the camera to take the image from
	camera = np.random.choice(['center', 'left', 'right'])

    # adjust the steering angle for left anf right cameras
	if camera == 'left':
		steering += correction
		imgPath = line[1]
	elif camera == 'right':
		steering -= correction
		imgPath = line[2]

	img = cv2.imread(imgPath)
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


    # decide whether to horizontally flip the image:
    # This is done to reduce the bias for turning left that is present in the training data
	flip_prob = np.random.random()
	if flip_prob > 0.5:
		    # flip the image and reverse the steering angle
		steering = -1*steering
		img= cv2.flip(img, 1)

	img = image_preprocessing(img)
	"""
	# Flipping the image so the network is trained to move in the mirror imaged path too.
	# This also help generate more training data.
	flippedImg = np.fliplr(img)
	# add the original image and the flipped image to the list of camera images.
	cameraImages.append(img)
	cameraImages.append(flippedImg)
	"""


	"""
	steeringLeft = steeringCenter + correction
	steeringRight = steeringCenter - correction
	# measurement of steering angle corresponding to the flipped img.
	measurementFlipped = -steeringCenter
	# append the steering measurement for the original image and the flipped image.
	steeringMeasurements.append(steeringCenter)
	steeringMeasurements.append(measurementFlipped)

	# append the left camera image with the correction added to the streering measurement.

	leftImgPath = line[1]
	leftImg = cv2.imread(leftImgPath)
	leftImg = cv2.cvtColor(leftImg, cv2.COLOR_BGR2RGB)

	rightImgPath = line[2]
	rightImg = cv2.imread(rightImgPath)
	rightImg = cv2.cvtColor(rightImg, cv2.COLOR_BGR2RGB)


	cameraImages.append(leftImg)
	cameraImages.append(rightImg)

	steeringMeasurements.append(steeringLeft)
	steeringMeasurements.append(steeringRight)
	"""

	return img, steering

def data_generator(csv_lines, batch_size=64):
 # Create empty arrays to contain batch of features and labels#
	X_batch = np.zeros((batch_size, 64, 64, 3), dtype=np.float32)
	y_batch = np.zeros((batch_size,1), dtype=np.float32)
	# will be generating 4 images per line of csv.
	# center camera image + its flipped image, and right and left cameras.
	N = len(csv_lines)
	no_batches_per_epoch = (N // batch_size)
	total_count = 0
	while True:
		for j in range(batch_size):
		# choose random index in features.
			X_batch[j], y_batch[j] = process_get_data(csv_lines[total_count + j])

		total_count = total_count + batch_size
		if total_count >= no_batches_per_epoch - 1:
            # reset the index, this allows iterating though the dataset again.
			total_count = 0

		yield X_batch, y_batch



def image_preprocessing(image):
	"""
	Crop the image, resize and normalize.
	"""
	image = image[50:130, :, :]
	image = cv2.resize(image, (64, 64))
	image = image.astype(np.float32)
	image = image/255.0 - 0.5
	return image

def get_model():
	model = Sequential()

	# Convolution Layer 1.
	model.add(Convolution2D(32, 5, 5, input_shape=(64, 64, 3), activation='elu'))

	#  Convolution Layer 2.
	model.add(Convolution2D(16, 3, 3, activation='elu'))

	model.add(Dropout(.25))
	model.add(MaxPooling2D((2, 2)))

	# Convolution Layer 3.
	model.add(Convolution2D(8, 3, 3, activation='elu'))

	model.add(Dropout(.25))

	# Flatten the output
	model.add(Flatten())

	# layer 4
	model.add(Dense(1024))
	model.add(Dropout(.3))
	model.add(ELU())

	# layer 5
	model.add(Dense(256))
	model.add(ELU())


	# Finally a single output, since this is a regression problem
	model.add(Dense(1))

	model.compile(optimizer="adam", loss="mse")

	return model


# Data read from csv. CSV contains the image path and steerig angles recorded at various points of gameplay.
lines = []

csvPath = "/Users/hackintoshrao/Documents/code/self-drive/driving_log.csv"


with open(csvPath) as csvData:
	# read from csv
	reader = csv.reader(csvData)
	for line in reader:
		lines.append(line)

# this is a parameter to tune
correction = 0.25

# shuffling the inputs.
random.shuffle(lines)

 # 80% of the data is used for training and 20% for validation.
training_split = 0.8

training_set_num = int(len(lines) * training_split)

training_data = lines[0:training_set_num-1]
validation_data = lines[training_set_num:]

print("training len: ", len(training_data))
print("validation len: ", len(validation_data))

BATCH_SIZE = 64

training_data_generator = data_generator(training_data, batch_size=BATCH_SIZE)
validation_data_generator = data_generator(validation_data, batch_size=BATCH_SIZE)

model = get_model()

samples_per_epoch = (len(lines)//BATCH_SIZE) * BATCH_SIZE

#lines = None

model.fit_generator(training_data_generator, validation_data=validation_data_generator,samples_per_epoch=samples_per_epoch, nb_epoch=3, nb_val_samples=3000)

print("Saving model weights and configuration file.")

model.save('model_all_image.h5')  # always save your weights after training or during training
with open('model.json', 'w') as outfile:
    outfile.write(model.to_json())

"""
for line in lines:
	imgPath = line[0]
	img = cv2.imread(imgPath)
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	# Flipping the image so the network is trained to move in the mirror imaged path too.
	# This also help generate more training data.
	flippedImg = np.fliplr(img)
	# add the original image and the flipped image to the list of camera images.
	cameraImages.append(img)
	cameraImages.append(flippedImg)

	steeringCenter = float(line[3])

	steeringLeft = steeringCenter + correction
	steeringRight = steeringCenter - correction
	# measurement of steering angle corresponding to the flipped img.
	measurementFlipped = -steeringCenter
	# append the steering measurement for the original image and the flipped image.
	steeringMeasurements.append(steeringCenter)
	steeringMeasurements.append(measurementFlipped)

	# append the left camera image with the correction added to the streering measurement.

	leftImgPath = line[1]
	leftImg = cv2.imread(leftImgPath)
	leftImg = cv2.cvtColor(leftImg, cv2.COLOR_BGR2RGB)

	rightImgPath = line[2]
	rightImg = cv2.imread(rightImgPath)
	rightImg = cv2.cvtColor(rightImg, cv2.COLOR_BGR2RGB)


	cameraImages.append(leftImg)
	cameraImages.append(rightImg)

	steeringMeasurements.append(steeringLeft)
	steeringMeasurements.append(steeringRight)
	"""

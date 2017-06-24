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


def add_full_path(path):
	"""
	Obtain the absolute path of the image, the CSV contains only the relative paths of the image.
	"""
	return '/Users/hackintoshrao/mycode/go/src/github.com/hackintoshrao/behavioral-cloning/data/' + path.strip()
	#return  path.strip()


def get_image(path):
	"""
	Read the image from its path and convert it to RGB and return.
	"""
	img = cv2.imread(path)
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	return img

def augment_brightness_camera_images(image):
    '''
    :param image: Input image
    :return: output image with reduced brightness
    '''

    # convert to HSV so that its easy to adjust brightness
    image1 = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)

    # randomly generate the brightness reduction factor
    # Add a constant so that it prevents the image from being completely dark
    random_bright = .25+np.random.uniform()

    # Apply the brightness reduction to the V channel
    image1[:,:,2] = image1[:,:,2]*random_bright

    # convert to RBG again
    image1 = cv2.cvtColor(image1,cv2.COLOR_HSV2RGB)
    return image1

def process_get_data(line):
	"""
	Randomly choose Center/Left/Right image -> Read the image and steering values -> pre-process image
	-> flip the image -> return both the original and the flipped image.
	"""
	"""
	Randomly choose Center/Left/Right image -> Read the image and steering values -> pre-process image
	-> flip the image -> return both the original and the flipped image.
	"""
	imgPath = line[0]
	imgPath = add_full_path(imgPath)

	steering = float(line[3])
	# randomly choose the camera to take the image from
	camera = np.random.choice(['center', 'left', 'right'])

	# adjust the steering angle for left anf right cameras
	if camera == 'right':
		steering_right = steering - correction
		imgPath_right = line[2]
		imgPath_right = add_full_path(imgPath_right)
		imgPath = imgPath_right
		steering = steering_right

	elif camera == 'left':
		steering_left = steering + correction
		imgPath_left = line[1]
		imgPath_left = add_full_path(imgPath_left)
		imgPath = imgPath_left
		steering = steering_left

	img = get_image(imgPath)

	flip_prob = np.random.random()
	#if flip_prob > 0.5:
        # flip the image and reverse the steering angle
	steering_flip = -1*steering
	img_flipped = cv2.flip(img, 1)

	img = augment_brightness_camera_images(img)
	img_flipped = augment_brightness_camera_images(img_flipped)
	#steering_flipped = -1*steering
	#img_flipped = cv2.flip(img, 1)
	img = image_preprocessing(img)
	img_flipped = image_preprocessing(img_flipped)

	return img, steering, img_flipped, steering_flip

def data_generator(csv_lines, batch_size=64):
	"""
	data generator, which is used to obtain the traning an validation data in batches
	while training the model
	"""
 # Create empty arrays to contain batch of features and labels#
	X_batch = np.zeros((batch_size, 64, 64, 3), dtype=np.float32)
	y_batch = np.zeros((batch_size,1), dtype=np.float32)
	# will be generating 4 images per line of csv.
	# center camera image + its flipped image, and right and left cameras.
	N = len(csv_lines)
	no_batches_per_epoch = (N // batch_size)
	total_count = 0
	while True:
		for j in range(batch_size//2):
			k = 2*j
		# choose random index in features.
			X_batch[k], y_batch[k], X_batch[k+1], y_batch[k+1] = process_get_data(csv_lines[total_count + j])

		total_count = total_count + batch_size//2
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
	"""
	Obtain the convolutional neural network model
	The model contains 3 convolutional layer and 2 fully connected layer.
	ELU is used as activation function.
	"""
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
	model.add(Dense(512))
	model.add(Dropout(.2))
	model.add(ELU())

	# layer 6
	model.add(Dense(256))
	model.add(ELU())

	# layer 7
	model.add(Dense(128))
	model.add(ELU())


	# Finally a single output, since this is a regression problem
	model.add(Dense(1))

	model.compile(optimizer="adam", loss="mse")

	return model


# Data read from csv. CSV contains the image path and steerig angles recorded at various points of gameplay.
lines = []

csvPath = "./data/driving_log.csv"
#csvPath = "/Users/hackintoshrao/Documents/code/self-drive/driving_log.csv"


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

# separate training and validation data and create different generators for them.
training_data = lines[0:training_set_num-1]
validation_data = lines[training_set_num:]

print("training len: ", len(training_data))
print("validation len: ", len(validation_data))

BATCH_SIZE = 32

# obtain generators for training and validation data.
training_data_generator = data_generator(training_data, batch_size=BATCH_SIZE)
validation_data_generator = data_generator(validation_data, batch_size=BATCH_SIZE)

# fetch the model.
model = get_model()

 # extracts around 22000 samples in each epoch from the generator.
samples_per_epoch = (20000//BATCH_SIZE) * BATCH_SIZE


model.fit_generator(training_data_generator, validation_data=validation_data_generator,samples_per_epoch=samples_per_epoch, nb_epoch=3, nb_val_samples=3000)

print("Saving model.")

model.save('model_udacity_6.h5')  # always save your weights after training or during training

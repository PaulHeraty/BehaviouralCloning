#!/usr/bin/python3

# This file generates a Keras mode (model.json) and a corresponding
# weights file (model.h5) which are used to implement behavioral cloning
# for driving a car around a race track. The model takes input frames
# (640x480x3) and labels which contain the steering angle for each frame.
# The model should then be able to predict a steering angle when presented
# which a previously un-seen frame. This can then be used to calculate how
# to steer a car on a track in order to stay on the road

################################################################
# Start by importing the required libraries
################################################################
import numpy as np
from keras.models import Sequential
from keras.layers import Convolution2D, Flatten, MaxPooling2D
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import Adam
from keras.callbacks import Callback
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt
from random import shuffle
import scipy.stats as stats
import pylab as pl
import os
import cv2
import csv
import math
import json
from pandas.stats.moments import ewma
from keras.models import model_from_json


################################################################
# Define our variables here
################################################################
fine_tune_mode = True
size_image = True
use_shuffle = True

use_dropout = True
dropout_factor = 0.4
w_reg=0.00
batch_norm = False
norm_inputs = True
max_pool = False
if fine_tune_mode:
    #learning_rate = 0.000001  
    learning_rate = 0.000001  
    use_3cams = False
    use_flip = False
else:
    learning_rate = 0.002  
    use_3cams = False
    use_flip = True

image_sizeX = 320
image_sizeY = 160
if size_image:
    image_sizeX = 160
    image_sizeY = 80
num_channels = 3 
n_classes = 1 # This is a regression, not a classification
nb_epoch = 5
batch_size = 100

input_shape1 = (image_sizeY, image_sizeX, num_channels)
num_filters1 = 24
filter_size1 = 5
stride1=(2,2)
num_filters2 = 36
filter_size2 = 5
stride2=(2,2)
num_filters3 = 48
filter_size3 = 5
stride3=(2,2)
num_filters4 = 64
filter_size4 = 3
stride4=(1,1)
num_filters5 = 64
filter_size5 = 3
stride5=(1,1)
pool_size = (2, 2)
hidden_layers1 = 100
hidden_layers2 = 50

################################################################
# Define any functions that we need
################################################################

# Read in the image, flip in necessary
def process_image(filename, flip=0):
    #print("Reading image file {}".format(filename))
    image = cv2.imread(filename)
    if size_image:
        image = cv2.resize(image, (image_sizeX, image_sizeY))
    #print("Image read...")
    if flip == 1:
        image = cv2.flip(image, 1)
        #print("Image flipped...")
    # Normalization now done in graph
    #normalized_image = normalize_image(image)
    normalized_image = image
    #print("Addding axis..")
    final_image = normalized_image[np.newaxis, ...]
    #print("Returning image...")
    return final_image

# Routine to plot a graph showing angle distributions
def plot_dist(X):
    y_train_list = sorted([x[1] for x in X])
    y_train_list = [float(i) for i in y_train_list]
    fit = stats.norm.pdf(y_train_list, np.mean(y_train_list), np.std(y_train_list))    
    pl.plot(y_train_list,fit,'-o')
    pl.hist(y_train_list,normed=True)
    pl.show()

# Plot the steering angles over time
def plot_steering(X, X_smooth):
    y = [x[1] for x in X]
    y_smooth = [x[1] for x in X_smooth]
    x = [i for i in range(len(X))]
    plt.plot(x, y, label='orig')
    plt.plot(x, y_smooth, label='smooth')
    plt.xlabel('Frame Number')
    plt.ylabel('Steering Angle')
    plt.legend(loc='upper right')
    plt.show()

# Calculate the correct number of samples per epoch based on batch size
def calc_samples_per_epoch(array_size, batch_size):
    num_batches = array_size / batch_size
    # return value must be a number than can be divided by batch_size
    samples_per_epoch = math.ceil((num_batches / batch_size) * batch_size)
    samples_per_epoch = samples_per_epoch * batch_size
    return samples_per_epoch
    
# Import the training data
# Note: the training image data is stored in the IMG directory, and 
# are 640x480 RGB images. Since there will likely be thousands of these
# images, we'll need to use Python generators to access these, thus
# preventing us from running out of memory (which would happen if I 
# tried to store the entire set of images in memory as a list

def get_next_image_angle_pair(image_list):
    index = 0
    #print("Len : {}".format(len(image_list)))
    while 1:
        final_images = np.ndarray(shape=(batch_size, image_sizeY, image_sizeX, num_channels), dtype=float)
        final_angles = np.ndarray(shape=(batch_size), dtype=float)
        for i in range(batch_size):
            if index >= len(image_list):
                index = 0
                if use_shuffle:
                    shuffle(image_list)
            filename = image_list[index][0]
            #print("Grabbing image {}".format(filename))
            angle = image_list[index][1]
            #print("  Angle {}".format(angle))
            flip = image_list[index][2]
            #print("  Flip {}".format(flip))
            final_image = process_image(filename, flip)
            #print("Processed image {}".format(filename))
            final_angle = np.ndarray(shape=(1), dtype=float)
            final_angle[0] = angle
            final_images[i] = final_image
            final_angles[i] = angle
            index += 1
        #print("Returning next batch")
        yield ({'batchnormalization_input_1' : final_images}, {'output' : final_angles})
        #yield ({'convolution2d_input_1' : final_images}, {'output' : final_angles})

###############################################
############### START #########################
###############################################

# Start by reading in the .csv file which has the filenames and steering angles
# driving_log_list is a list of lists, where element [x][0] is the image file name
# and element [x][3] is the steering angle
with open('driving_log.csv', 'r') as f:
    reader = csv.reader(f)
    driving_log_list = list(reader)
num_frames = len(driving_log_list)
#num_frames=500
print("Found {} frames of input data.".format(num_frames))

# Process this list so that we end up with training images and labels
if use_3cams:
    X_train = [("", 0.0, 0) for x in range(num_frames*3)]
    print(len(X_train))
    for i in range(num_frames):
        #print(i)
        X_train[i*3] = (driving_log_list[i][0].lstrip(),         # center image
                  float(driving_log_list[i][3]),  # center angle 
                  0)                              # dont flip
        X_train[(i*3)+1] = (driving_log_list[i][1].lstrip(),       # left image
                  float(driving_log_list[i][3]) + 0.08,  # left angle 
                  0)                              # dont flip
        X_train[(i*3)+2] = (driving_log_list[i][2].lstrip(),         # right image
                  float(driving_log_list[i][3]) - 0.08,  # right angle 
                  0)                              # dont flip
else:
    X_train = [("", 0.0, 0) for x in range(num_frames)]
    print(len(X_train))
    for i in range(num_frames):
        #print(i)
        X_train[i] = (driving_log_list[i][0].lstrip(),         # center image
                  float(driving_log_list[i][3]),  # center angle 
                  0)                              # dont flip

# Update num_frames as needed
num_frames = len(X_train)

# Also, in order to generate more samples, lets add entries twice for
# entries that have non-zero angles, and add a flip switch. Then when
# we are reading these, we will flip the image horizontally and 
# negate the angles
if use_flip:
    for i in range(num_frames):
        if X_train[i][1] != 0.0:
            X_train.append([X_train[i][0], -1.0 * X_train[i][1], 1]) # flip flag

num_frames = len(X_train)
print("After list pre-processing, now have {} frames".format(num_frames))

# Split some of the training data into a validation dataset.
# First lets shuffle the dataset, as we added lots of non-zero elements to the end
if use_shuffle:
    shuffle(X_train)
num_train_elements = int((num_frames/4.)*3.)
num_valid_elements = int(((num_frames/4.)*1.) / 2.)
X_valid = X_train[num_train_elements:num_train_elements + num_valid_elements]
X_test = X_train[num_train_elements + num_valid_elements:]
X_train = X_train[:num_train_elements]
print("X_train has {} elements.".format(len(X_train)))
print("X_valid has {} elements.".format(len(X_valid)))
print("X_test has {} elements.".format(len(X_test)))

# Lets look at the distribution of items
#plot_dist(X_train)

################################################################
# Load the existing model  & weights if we are fine tuning
################################################################
if fine_tune_mode:
    print("**********************************")
    print("*** Running in FINE-TUNE mode! ***")
    print("**********************************")
    with open("model.json.save", 'r') as jfile:
        model = model_from_json(json.load(jfile))

    model.compile("adam", "mse")
    weights_file = "model.h5.save"
    model.load_weights(weights_file)
else:
################################################################
# Otherwise build a new CNN Network with Keras
################################################################
    print("**********************************")
    print("*** Running in NEW MODEL mode! ***")
    print("**********************************")
    model = Sequential()
    # CNN Layer 1
    if norm_inputs:
        model.add(Lambda(lambda x: x/128. -1.,
                        input_shape=input_shape1,
                        output_shape=input_shape1))
        #model.add(BatchNormalization(input_shape=input_shape1, axis=1))
    model.add(Convolution2D(nb_filter=num_filters1, 
                        nb_row=filter_size1, 
                        nb_col=filter_size1,
                        subsample=stride1,
                        border_mode='valid',
                        input_shape=input_shape1, 
                        W_regularizer=l2(w_reg)))
    if max_pool:
        model.add(MaxPooling2D(pool_size=pool_size))
    if batch_norm:
        model.add(BatchNormalization(axis=1))
    model.add(Activation('relu'))
    # CNN Layer 2
    model.add(Convolution2D(nb_filter=num_filters2, 
                        nb_row=filter_size2, 
                        nb_col=filter_size2,
                        subsample=stride2,
                        border_mode='valid', 
                        W_regularizer=l2(w_reg)))
    if max_pool:
        model.add(MaxPooling2D(pool_size=pool_size))
    if use_dropout:
        model.add(Dropout(dropout_factor))
    if batch_norm:
        model.add(BatchNormalization(axis=1))
    model.add(Activation('relu'))
    # CNN Layer 3
    model.add(Convolution2D(nb_filter=num_filters3, 
                        nb_row=filter_size3, 
                        nb_col=filter_size3,
                        subsample=stride3,
                        border_mode='valid', 
                        W_regularizer=l2(w_reg)))
    if max_pool:
        model.add(MaxPooling2D(pool_size=pool_size))
    if use_dropout:
        model.add(Dropout(dropout_factor))
    if batch_norm:
        model.add(BatchNormalization(axis=1))
    model.add(Activation('relu'))
    # CNN Layer 4
    model.add(Convolution2D(nb_filter=num_filters4,
                        nb_row=filter_size4, 
                        nb_col=filter_size4,
                        subsample=stride4,
                        border_mode='valid', 
                        W_regularizer=l2(w_reg)))
    if max_pool:
        model.add(MaxPooling2D(pool_size=pool_size))
    if use_dropout:
        model.add(Dropout(dropout_factor))
    if batch_norm:
        model.add(BatchNormalization(axis=1))
    model.add(Activation('relu'))
    # CNN Layer 5
    model.add(Convolution2D(nb_filter=num_filters5,
                        nb_row=filter_size5, 
                        nb_col=filter_size5,
                        subsample=stride5,
                        border_mode='valid', 
                        W_regularizer=l2(w_reg)))
    if max_pool:
        model.add(MaxPooling2D(pool_size=pool_size))
    if use_dropout:
        model.add(Dropout(dropout_factor))
    if batch_norm:
        model.add(BatchNormalization(axis=1))
    model.add(Activation('relu'))
    # Flatten
    model.add(Flatten())
    # FCNN Layer 1
    #model.add(Dense(hidden_layers1, input_shape=(7200,), name="hidden1", W_regularizer=l2(w_reg)))
    if size_image:
        model.add(Dense(hidden_layers1, input_shape=(2496,), name="hidden1", W_regularizer=l2(w_reg)))
    else:
        model.add(Dense(hidden_layers1, input_shape=(27456,), name="hidden1", W_regularizer=l2(w_reg)))
    model.add(Activation('relu'))
    # FCNN Layer 2
    model.add(Dense(hidden_layers2, name="hidden2", W_regularizer=l2(w_reg)))
    model.add(Activation('relu'))
    # FCNN Layer 3
    model.add(Dense(n_classes, name="output", W_regularizer=l2(w_reg)))

#model.summary()

################################################################
# Train the network using generators
################################################################
print("Number of Epochs : {}".format(nb_epoch))
print("  Batch Size : {}".format(batch_size))
print("  Training batches : {} ".format(calc_samples_per_epoch(len(X_train), batch_size)))
print("  Validation batches : {} ".format(calc_samples_per_epoch(len(X_valid), batch_size)))

if fine_tune_mode:
    print("*** Fine-tuning model with learning rate {} ***".format(learning_rate))
else:
    print("*** Compiling new model wth learning rate {} ***".format(learning_rate))

model.compile(loss='mean_squared_error',
              optimizer=Adam(lr=learning_rate)
              )

history = model.fit_generator(
                    get_next_image_angle_pair(X_train),	# The generator to return batches to train on
                    nb_epoch=nb_epoch,  		# The number of epochs we will run for
                    max_q_size=10,      		# Max generator items that are queued and ready 
                    samples_per_epoch=calc_samples_per_epoch(len(X_train), batch_size),
                    validation_data=get_next_image_angle_pair(X_valid),	# validation data generator
                    nb_val_samples=calc_samples_per_epoch(len(X_valid), batch_size),
                    verbose=1)

# Evaluate the accuracy of the model using the test set
score = model.evaluate_generator(
                    generator=get_next_image_angle_pair(X_test),	# validation data generator
                    val_samples=calc_samples_per_epoch(len(X_test), batch_size), # How many batches to run in one epoch
                    )
print("Test score {}".format(score))

################################################################
# Save the model and weights
################################################################
model_json = model.to_json()
with open("./model.json", "w") as json_file:
    #json_file.write(model_json)
    json.dump(model_json, json_file)
model.save_weights("./model.h5")
print("Saved model to disk")

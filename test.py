#!/home/pedgrfx/anaconda3/bin/python
import argparse
import base64
import json
import csv
import cv2

import numpy as np
import socketio
import eventlet
import eventlet.wsgi
import time
from PIL import Image
from PIL import ImageOps
from flask import Flask, render_template
from io import BytesIO

from keras.models import model_from_json
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array
import matplotlib.pyplot as plt

with open("model.json", 'r') as jfile:
    model = model_from_json(json.load(jfile))

model.compile("adam", "mse")
weights_file = "model.h5"
model.load_weights(weights_file)

with open('driving_log.csv', 'r') as f:
    reader = csv.reader(f)
    driving_log_list = list(reader)
num_frames = len(driving_log_list)

X_train = [("",0.,0) for x in range(num_frames)]
for i in range(num_frames):
    X_train[i] = (driving_log_list[i][0], float(driving_log_list[i][3]), 0) # dont flip

x_test = X_train[27][0]  # Turning left
y_test = X_train[27][1]
x_test2 = X_train[10][0]  # Staying straight
y_test2 = X_train[10][1]
x_test3 = X_train[489][0]  # Turning right
y_test3 = X_train[489][1]
print("y_test value : {}".format(y_test))
print("y_test2 value : {}".format(y_test2))
print("y_test3 value : {}".format(y_test3))

def process_image(filename, flip=0):
    #print("Reading image file {}".format(filename))
    image = cv2.imread(filename)
    if flip == 1:
        image = cv2.flip(image, 1)
    normalized_image = image
    final_image = normalized_image[np.newaxis, ...]
    return final_image

# Test the model on an image and see how well it performs vs. the label
# Test 1 - turning left
test_image = process_image(x_test)
prediction = model.predict(
                    x = test_image,
                    batch_size=1,
                    verbose=1)
# Need to reshape processed image for display
test_image = test_image.reshape(test_image.shape[1:])
print("Model predicted {} for image which had value of {}".format(prediction, y_test))
plt.imshow(test_image)
plt.show()

# Test 2 -  driving straight
test_image2 = process_image(x_test2)
prediction2 = model.predict(
                    x = test_image2,
                    batch_size=1,
                    verbose=1)
# Need to reshape processed image for display
test_image2 = test_image2.reshape(test_image2.shape[1:])
print("Model predicted {} for image which had value of {}".format(prediction2, y_test2))
plt.imshow(test_image2)
plt.show()

# Test 3 -  turning right
test_image3 = process_image(x_test3)
prediction3 = model.predict(
                    x = test_image3,
                    batch_size=1,
                    verbose=1)
# Need to reshape processed image for display
test_image3 = test_image3.reshape(test_image3.shape[1:])
print("Model predicted {} for image which had value of {}".format(prediction3, y_test3))
plt.imshow(test_image3)
plt.show()

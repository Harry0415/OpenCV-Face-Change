# -*- coding: utf-8 -*-
"""
Created on Thu Jun 17 12:15:22 2021

@author: herry
"""

import mtcnn
from mtcnn.mtcnn import MTCNN
from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input
from keras_vggface.utils import decode_predictions
import PIL
import os
from urllib import request
import numpy as np
import cv2

# # Give the image link
# # url = "https://upload.wikimedia.org/wikipedia/commons/thumb/8/8d/Channing_Tatum_by_Gage_Skidmore_3.jpg/330px-Channing_Tatum_by_Gage_Skidmore_3.jpg"
# url = "https://upload.wikimedia.org/wikipedia/commons/3/33/Reuni%C3%A3o_com_o_ator_norte-americano_Keanu_Reeves_%2846806576944%29_%28cropped%29.jpg"
# # Open the link and save the image to res
# res = request.urlopen(url)
# # Read the res object and convert it to an array
# img = np.asarray(bytearray(res.read()), dtype='uint8')

# # Add the color variable
# img = cv2.imdecode(img, cv2.IMREAD_COLOR)
# print(img)
# cv2.imshow("img", img)
# cv2.waitKey()

# Load images and convert to gray scale
img = cv2.imread("../input/jim_carrey.jpg")
img = np.asarray(img, dtype='uint8')
cv2.imshow("123", img)
cv2.waitKey()


detector = MTCNN()

# set face extraction parameters
target_size = (224,224) # output image size
border_rel = 0 # increase or decrease zoom on image

# detect faces in the image
detections = detector.detect_faces(img)
print(detections)

x1, y1, width, height = detections[0]['box']
dw = round(width * border_rel)
dh = round(height * border_rel)
x2, y2 = x1 + width + dw, y1 + height + dh
face = img[y1:y2, x1:x2]

# resize pixels to the model size
face = PIL.Image.fromarray(face)
face = face.resize((224, 224))
face = np.asarray(face)
# show face
# cv2.imshow("face", face)

# convert to float32
face_pp = face.astype('float32')
face_pp = np.expand_dims(face_pp, axis = 0)

face_pp = preprocess_input(face_pp, version = 2)

# Create the resnet50 Model
model = VGGFace(model= 'resnet50')
# Check what the required input of the model is & output
print('Inputs: {input}'.format(input = model.inputs))
print('Output: {output}'.format(output = model.outputs))

# predict the face with the input
prediction = model.predict(face_pp)

# convert predictions into names & probabilities
results = decode_predictions(prediction)
# Display results
cv2.imshow("img", img)

for result in results[0]:
    print ('%s: %.3f%%' % (result[0], result[1]*100))
    
cv2.waitKey()
cv2.destroyAllWindows()

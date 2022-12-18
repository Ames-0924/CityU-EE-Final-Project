import cv2
import numpy as np
from keras.models import load_model
import os

# Load the model
model = load_model('keras_model.h5')

# CAMERA can be 0 or 1 based on default camera of your computer.
camera = cv2.VideoCapture(0)

# Grab the labels from the labels.txt file. This will be used later.
labels = open('labels.txt', 'r').readlines()
i = 0
while True:
    # Grab the webcameras image.
    ret, image = camera.read()
    # Resize the raw image into (224-height,224-width) pixels.
    # image = cv2.resize(image, (224, 336), interpolation=cv2.INTER_AREA)
    image = cv2.resize(image, (384, 224), interpolation=cv2.INTER_AREA)
    image = image[0:224, 80:304]
    # 150528
    # Show the image in a window
    cv2.imshow('Webcam Image', image)
    # Make the image a numpy array and reshape it to the models input shape.
    image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)
    # Normalize the image array
    image = (image / 127.5) - 1
    # Have the model predict what the current image is. Model.predict
    # returns an array of percentages. Example:[0.2,0.8] meaning its 20% sure
    # it is the first label and 80% sure its the second label.
    
    i = i + 1
    if i % 15 == 0:
        os.system('clear')
        probabilities = model.predict(image)
        keyboard_input = cv2.waitKey(1)
        # 27 is the ASCII for the esc key on your keyboard
        print(keyboard_input)
        print(probabilities)
        print(labels[np.argmax(probabilities)])
        if (np.argmax(probabilities) == 0 or np.argmax(probabilities)==2) or (probabilities[0][2] > 0.2):
            print("False")
        print('-')
        print('-')
        print('-')
        print('-')
        print('-')
        print('-')
        
    
    # Print what the highest value probabilitie label
    # print(labels[np.argmax(probabilities)])
    # Listen to the keyboard for presses.

   

camera.release()
cv2.destroyAllWindows()

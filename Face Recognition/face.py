import tensorflow.keras
from PIL import Image, ImageOps
import numpy as np
import os

np.set_printoptions(suppress=True)

# Load the model
model = tensorflow.keras.models.load_model(r'C:\Users\Gobu\OneDrive\Desktop\converted_keras (3)\keras_model.h5')

data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

def classify(x):
        image = Image.open(x)
        size = (224, 224)
        image = ImageOps.fit(image, size, Image.ANTIALIAS)
        image_array = np.asarray(image)
        normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
        data[0] = normalized_image_array
        prediction = model.predict(data)[0].tolist()
        print(prediction)

        if max(prediction) > 0.70:
                print("Access Granted")

        else:
                print("Access Rejected")

import cv2
camera = cv2.VideoCapture(0)
return_value, image = camera.read()
cv2.imwrite('FaceRec.jpg', image)
del(camera)

classify('FaceRec.jpg')


os.remove("FaceRec.jpg")

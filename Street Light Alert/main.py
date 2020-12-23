import tensorflow.keras
from PIL import Image, ImageOps
import numpy as np
from gtts import gTTS
import os
import playsound

def speak(text):
    tts = gTTS(text=text, lang='en')
    filename = 'voice.mp3'
    tts.save(filename)
    playsound.playsound(filename)

while True:
    # Disable scientific notation for clarity
    np.set_printoptions(suppress=True)

    path = input("Enter path of file - ")

    # Load the model
    model = tensorflow.keras.models.load_model(r'C:\Users\Gobu\OneDrive\Desktop\StreetLights\keras_model.h5')

    # Create the array of the right shape to feed into the keras model
    # The 'length' or number of images you can put into the array is
    # determined by the first position in the shape tuple, in this case 1.
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

    # Replace this with the path to your image
    image = Image.open(path)

    #resize the image to a 224x224 with the same strategy as in TM2:
    #resizing the image to be at least 224x224 and then cropping from the center
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.ANTIALIAS)

    #turn the image into a numpy array
    image_array = np.asarray(image)

    # display the resized image
    image.show()

    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1

    # Load the image into the array
    data[0] = normalized_image_array

    # run the inference
    prediction = model.predict(data).tolist()
    print(prediction)

    p = prediction[0].index(max(prediction[0]))

    if p==0:
        speak("the street light is red. Stop the car")
    elif p==1:
        speak("the street light is green. you may start driving the car")
    elif p==2:
        speak("the walking street light is red. Stop walking")
    elif p==3:
        speak("the walking street light is green. you may start walking now")

    ch = input("Do you want to continue? [y/n] :- ")
    if "n" in ch:
        print("Thanks for using this service")
        break

    os.remove("voice.mp3")

os.remove("voice.mp3")

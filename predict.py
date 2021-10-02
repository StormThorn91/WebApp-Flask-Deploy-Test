import numpy as np
import tensorflow as tf
import os
import sys
from PIL import Image
import skimage
from skimage import transform

sys.path.append(os.path.abspath('./model'))

#Initialize model
def init_model():
    loaded_model = tf.keras.models.load_model('model/classifier-model.h5')
    print("Loaded model successfully")
    return loaded_model

#Make a prediction based on the uploaded image
def predict(model, img):
    np_image = Image.open(img)
    np_image = np.array(np_image).astype('float32')/255
    np_image = transform.resize(np_image, (256, 256, 3))
    np_image = np.expand_dims(np_image, axis=0)

    CATEGORIES = ["Brown Spot", "Common Rust",
                  "Healthy", "Northern Leaf Blight"]
                  
    prediction = model.predict(np_image)
    prediction = np.argmax(prediction, axis=1)

    result = CATEGORIES[prediction[0]]

    return result
import tensorflow as tf
import numpy as np
import cv2

# Load model
model = tf.keras.models.load_model("model/weed_model.h5")

def predict_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (128, 128))
    image = image / 255.0
    image = np.expand_dims(image, axis=0)

    prediction = model.predict(image)[0][0]
    return "Weed" if prediction > 0.5 else "Crop"

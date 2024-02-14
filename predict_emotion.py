import cv2
import numpy as np
import tensorflow as tf
from PIL import Image as PilImage

checkpoint_path = 'C:/Users/dulsh/Documents/PROJECT_EUSL/pythonProject/model/best_model.h5'
model = tf.keras.models.load_model(checkpoint_path)

label_to_text = {0: 'anger', 1: 'disgust', 2: 'fear', 3: 'happiness', 4: 'sadness', 5: 'surprise', 6: 'neutral'}

def predict_expression(image):
    img = np.array(image)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img = cv2.resize(img, (48, 48))
    img = img.reshape(1, 48, 48, 1)
    img = img / 255.0

    prediction = model.predict(img)
    expression_label = label_to_text[np.argmax(prediction)]
    return expression_label

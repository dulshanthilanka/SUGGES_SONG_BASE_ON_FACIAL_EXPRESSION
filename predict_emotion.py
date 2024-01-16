import cv2
import os
import numpy as np
import tensorflow as tf

# Load the pre-trained model
checkpoint_path = 'C:/Users/dulsh/Desktop/facial_expression/checkpoint/best_model.h5'
model = tf.keras.models.load_model(checkpoint_path)

# Define emotion labels
label_to_text = {0: 'anger', 1: 'disgust', 2: 'fear', 3: 'happiness', 4: 'sadness', 5: 'surprise', 6: 'neutral'}

# Function to predict the expression for an image
def predict_expression(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (48, 48))
    img = img.reshape(1, 48, 48, 1)
    img = img / 255.0

    # Predict the expression using the model
    prediction = model.predict(img)
    expression_label = label_to_text[np.argmax(prediction)]
    return expression_label, img

# Path to the photo you want to analyze (change this to your photo's path)
photo_path = 'C:/Users/dulsh/Desktop/facial_expression/images/image1.jpg'

# Predict the expression and get the image
predicted_expression, image = predict_expression(photo_path)
print("Predicted Expression:", predicted_expression)

# Display the image with the predicted expression
image_display = image.squeeze()
image_display = (image_display * 255).astype(np.uint8)

cv2.putText(image_display, predicted_expression, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

# Use a different window name to avoid conflicts
cv2.namedWindow('Predicted Expression', cv2.WINDOW_NORMAL)
cv2.imshow('Predicted Expression', image_display)
cv2.waitKey(0)
cv2.destroyAllWindows()

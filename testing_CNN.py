import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

# Load the trained model
model = tf.keras.models.load_model("cnn_model.h5")
print("Model loaded successfully!")


# Function to display the image and make a prediction
def predict_and_display_image(image_path):
    try:
        # Load and preprocess the image
        img = tf.keras.utils.load_img(image_path, color_mode='grayscale', target_size=(128, 128))
        img_array = tf.keras.utils.img_to_array(img) / 255.0  # Normalize pixel values to [0, 1]
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension (1, 128, 128, 1)

        # Make a prediction
        prediction = model.predict(img_array)
        confidence = prediction[0][0]  # Raw prediction score
        result = "Healthy" if confidence < 0.5 else "Parkinson's Disease"

        # Display the image with the prediction and confidence
        plt.imshow(img, cmap='gray')
        plt.title(f"Prediction: {result} (Confidence: {confidence:.2f})")
        plt.axis('off')
        plt.show()

        print(f"Prediction for the image: {result}")
        print(f"Confidence Score: {confidence:.2f}")

    except Exception as e:
        print(f"Error: {e}")


# Get user input for the image path
while True:
    image_path = input("Please provide the full path to the .png image for testing (or type 'exit' to quit): ")
    if image_path.lower() == 'exit':
        break
    predict_and_display_image(image_path)

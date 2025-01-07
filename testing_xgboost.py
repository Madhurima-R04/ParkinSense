import cv2
import numpy as np
import os
import xgboost as xgb
from skimage.measure import shannon_entropy
from scipy.fftpack import fft
import matplotlib.pyplot as plt

#In feature extraction - edge detection, entropy, radius variation, fractal dimension

# Step 1: Define feature extraction function (same as during training)
def extract_features(image_path):
    # Load image in grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Error loading image: {image_path}")
        return None

    # Edge detection
    edges = cv2.Canny(image, threshold1=50, threshold2=150)

    # Calculate radius variation
    moments = cv2.moments(edges)
    if moments["m00"] != 0:
        cx = int(moments["m10"] / moments["m00"])
        cy = int(moments["m01"] / moments["m00"])
    else:
        cx, cy = 0, 0

    # Distance from centroid to edge points
    edge_points = np.argwhere(edges > 0)
    distances = np.sqrt((edge_points[:, 1] - cx) ** 2 + (edge_points[:, 0] - cy) ** 2)
    radius_mean = np.mean(distances) if len(distances) > 0 else 0
    radius_std = np.std(distances) if len(distances) > 0 else 0

    # Angle variance
    angles = np.arctan2(edge_points[:, 0] - cy, edge_points[:, 1] - cx)
    angle_std = np.std(angles) if len(angles) > 0 else 0

    # Additional features
    entropy = shannon_entropy(edges)
    freq_domain = fft(distances) if len(distances) > 0 else [0]
    freq_mean = np.mean(np.abs(freq_domain))
    freq_std = np.std(np.abs(freq_domain))

    # Fractal dimension (box counting)
    box_counts = []
    sizes = [2, 4, 8, 16, 32, 64]
    for size in sizes:
        resized = cv2.resize(edges, (size, size))
        box_counts.append(np.sum(resized > 0))
    fractal_dimension = -np.polyfit(np.log(sizes), np.log(box_counts), 1)[0]

    # Feature vector
    return [radius_mean, radius_std, angle_std, entropy, freq_mean, freq_std, fractal_dimension]


# Step 2: Load the trained model
bst = xgb.Booster()
bst.load_model("best_xgb_model_with_early_stopping.json")


# Step 3: Define function to classify an image and visualize it
def classify_image(image_path):
    # Extract features from the input image
    feature_vector = extract_features(image_path)

    if feature_vector is None:
        print("Error extracting features from the image.")
        return None

    # Prepare the feature vector for prediction
    dimage = xgb.DMatrix([feature_vector])

    # Predict the class (probability of being parkinson)
    y_pred = bst.predict(dimage)
    y_pred_binary = (y_pred > 0.5).astype(int)  # Convert probability to binary prediction

    # Map the prediction to the label
    prediction_label = "Healthy" if y_pred_binary == 0 else "Parkinson"
    confidence = y_pred[0] * 100 if y_pred_binary == 1 else (1 - y_pred[0]) * 100

    # Visualize the image with the prediction result
    image = cv2.imread(image_path)
    if image is not None:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB for Matplotlib
        plt.figure(figsize=(8, 6))
        plt.imshow(image)
        plt.title(f"Prediction: {prediction_label}\nConfidence: {confidence:.2f}%", fontsize=14)
        plt.axis("off")
        plt.show()
    else:
        print(f"Unable to load image for visualization: {image_path}")


# Step 4: Take image path as input from the user
image_path = input("Please enter the path to the image you want to classify: ")

# Step 5: Classify the input image
classify_image(image_path)

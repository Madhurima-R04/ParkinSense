import cv2
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns

# Step 1: Define the feature extraction function (same as during training)
def extract_features(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    edges = cv2.Canny(image, threshold1=50, threshold2=150)

    # Calculate radius variation (distance from centroid to edges)
    moments = cv2.moments(edges)
    if moments["m00"] != 0:
        cx = int(moments["m10"] / moments["m00"])
        cy = int(moments["m01"] / moments["m00"])
    else:
        cx, cy = 0, 0

    edge_points = np.argwhere(edges > 0)
    distances = np.sqrt((edge_points[:, 1] - cx) ** 2 + (edge_points[:, 0] - cy) ** 2)
    radius_mean = np.mean(distances)
    radius_std = np.std(distances)

    # Line smoothness (angle variance)
    angles = np.arctan2(edge_points[:, 0] - cy, edge_points[:, 1] - cx)
    angle_std = np.std(angles)

    # Additional Features: Edge Density and Histogram of Oriented Gradients (HOG)
    edge_density = np.sum(edges) / edges.size
    hog_descriptor = cv2.HOGDescriptor()
    hog_features = hog_descriptor.compute(image).flatten()[:32]  # Extract the first 32 HOG features

    # Combine all features
    return [radius_mean, radius_std, angle_std, edge_density] + hog_features.tolist()

# Step 2: Define a function to load data and extract features for testing
def load_data_and_extract_features(directory):
    features = []
    labels = []
    file_paths = []

    for label, category in enumerate(['healthy', 'parkinson']):
        category_path = os.path.join(directory, category)
        for filename in os.listdir(category_path):
            image_path = os.path.join(category_path, filename)
            feature_vector = extract_features(image_path)
            features.append(feature_vector)
            labels.append(label)
            file_paths.append(image_path)

    return np.array(features), np.array(labels), file_paths

# Step 3: Load the saved ensemble model
ensemble_model = joblib.load('ensemble_model.pkl')

# Specify the directory for testing images
test_dir = "archive/drawings/spiral/testing"

# Load testing data and features
X_test, y_test, file_paths = load_data_and_extract_features(test_dir)

# Step 4: Normalize the test data (use the same scaler as during training)
scaler = StandardScaler()
X_test = scaler.fit_transform(X_test)

# Apply PCA for dimensionality reduction (use the same PCA as during training)
pca = PCA(n_components=0.90, random_state=42)
X_test = pca.fit_transform(X_test)

# Step 5: Make predictions
y_pred = ensemble_model.predict(X_test)

# Step 6: Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Test accuracy of ensemble model: {accuracy * 100:.2f}%")

# Step 7: Plot the first few test images with predicted labels
num_images_to_show = 5  # You can change this value

plt.figure(figsize=(15, 10))
for i in range(num_images_to_show):
    image = cv2.imread(file_paths[i])
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB for displaying with matplotlib

    plt.subplot(2, num_images_to_show//2, i+1)
    plt.imshow(image)
    true_label = 'Healthy' if y_test[i] == 0 else 'Parkinson'
    predicted_label = 'Healthy' if y_pred[i] == 0 else 'Parkinson'
    plt.title(f"True: {true_label}\nPred: {predicted_label}")
    plt.axis('off')

plt.tight_layout()
plt.show()

# Step 8: Confusion Matrix Visualization without Seaborn
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 6))
plt.imshow(cm, interpolation='nearest', cmap='Blues')
plt.title('Confusion Matrix')
plt.colorbar()

tick_marks = np.arange(len(['Healthy', 'Parkinson']))
plt.xticks(tick_marks, ['Healthy', 'Parkinson'])
plt.yticks(tick_marks, ['Healthy', 'Parkinson'])

# Add text annotations
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")

plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.show()


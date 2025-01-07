import cv2
import numpy as np
import os
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
import joblib  # Import joblib for saving the model

# Step 1: Define an enhanced feature extraction function
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

# Step 2: Load data and extract features
def load_data_and_extract_features(directory):
    features = []
    labels = []

    for label, category in enumerate(['healthy', 'parkinson']):
        category_path = os.path.join(directory, category)
        for filename in os.listdir(category_path):
            image_path = os.path.join(category_path, filename)
            feature_vector = extract_features(image_path)
            features.append(feature_vector)
            labels.append(label)

    return np.array(features), np.array(labels)

# Specify directory paths
train_dir = "archive/drawings/spiral/training"
test_dir = "archive/drawings/spiral/testing"

# Load training and testing data
X_train, y_train = load_data_and_extract_features(train_dir)
X_test, y_test = load_data_and_extract_features(test_dir)

# Step 3: Normalize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Optional: Apply PCA for dimensionality reduction
pca = PCA(n_components=0.90, random_state=42)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)

# Step 4: Set up ensemble of models
svm_model = SVC(C=100, kernel='linear', gamma='scale', degree=2, probability=True)
rf_model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
knn_model = KNeighborsClassifier(n_neighbors=5)

# Combine models in a voting ensemble
ensemble_model = VotingClassifier(
    estimators=[('svm', svm_model), ('rf', rf_model), ('knn', knn_model)],
    voting='soft'
)

# Train the ensemble model
ensemble_model.fit(X_train, y_train)

# Step 5: Evaluate the model
y_pred = ensemble_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Test accuracy of ensemble model: {accuracy * 100:.2f}%")

# Save the ensemble model
joblib.dump(ensemble_model, 'ensemble_model.pkl')
print("Ensemble model saved as ensemble_model.pkl")

loaded_model = joblib.load('ensemble_model.pkl')

import cv2
import numpy as np
import os
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from skimage.measure import shannon_entropy
from scipy.fftpack import fft
import joblib

# Step 1: Define feature extraction function
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


# Step 2: Load data and extract features
def load_data_and_extract_features(directory):
    features = []
    labels = []

    for label, category in enumerate(['healthy', 'parkinson']):
        category_path = os.path.join(directory, category)
        if not os.path.isdir(category_path):
            print(f"Directory not found: {category_path}")
            continue

        for filename in os.listdir(category_path):
            image_path = os.path.join(category_path, filename)
            feature_vector = extract_features(image_path)
            if feature_vector is not None:  # Skip images that failed to load
                features.append(feature_vector)
                labels.append(label)

    return np.array(features), np.array(labels)


# Directory paths
train_dir = "archive/drawings/spiral/training"
test_dir = "archive/drawings/spiral/testing"

# Load training and testing data
X_train, y_train = load_data_and_extract_features(train_dir)
X_test, y_test = load_data_and_extract_features(test_dir)

# Check if data loading was successful
if X_train.size == 0 or X_test.size == 0:
    raise ValueError("Data loading failed. Check file paths and ensure that images are available.")

# Split training data for validation
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Step 3: Prepare the data in DMatrix format for xgb.train()
dtrain = xgb.DMatrix(X_train, label=y_train)
dval = xgb.DMatrix(X_val, label=y_val)
dtest = xgb.DMatrix(X_test, label=y_test)

# Step 4: Set parameters for training
params = {
    'max_depth': 5,
    'learning_rate': 0.1,
    'n_estimators': 500,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
}

# Step 5: Define evals for early stopping
evals = [(dtrain, 'train'), (dval, 'eval')]

# Step 6: Train the model using xgb.train() with early stopping
num_round = 500
early_stopping_rounds = 20

bst = xgb.train(
    params,
    dtrain,
    num_round,
    evals=evals,
    early_stopping_rounds=early_stopping_rounds,
    verbose_eval=True
)

# Step 7: Evaluate on test set
y_pred = bst.predict(dtest)
y_pred_binary = (y_pred > 0.5).astype(int)  # Convert probability to binary prediction
accuracy = accuracy_score(y_test, y_pred_binary)

print(f"Test accuracy: {accuracy * 100:.2f}%")

# Step 8: Save the trained model
bst.save_model("best_xgb_model_with_early_stopping.json")
print("Model saved as 'best_xgb_model_with_early_stopping.json'")

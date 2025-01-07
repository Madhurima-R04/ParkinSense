import cv2
import os


# Define base input and output paths
input_base_path = 'archive/drawings'
output_base_path = 'preprocessed_drawings'

# Define categories and subcategories
categories = ['spiral', 'wave']
subcategories = ['testing', 'training']
classes = ['healthy', 'parkinson']

# Ensure output base directory exists
os.makedirs(output_base_path, exist_ok=True)


# Preprocessing function
def preprocess_image(image_path, target_size=(128, 128)):
    # Load image
    image = cv2.imread(image_path)

    # Resize and convert to grayscale
    image = cv2.resize(image, target_size)
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Edge Detection
    edges = cv2.Canny(image, threshold1=50, threshold2=150)

    # Noise Reduction
    smoothed_image = cv2.GaussianBlur(edges, (5, 5), 0)

    return smoothed_image


# Process each category, subcategory, and class
for category in categories:
    for subcategory in subcategories:
        for class_type in classes:
            # Define paths
            input_path = os.path.join(input_base_path, category, subcategory, class_type)
            output_path = os.path.join(output_base_path, category, subcategory, class_type)
            os.makedirs(output_path, exist_ok=True)

            # Process each image in the class folder
            for image_name in os.listdir(input_path):
                image_path = os.path.join(input_path, image_name)

                # Preprocess image
                preprocessed_image = preprocess_image(image_path)

                # Save the preprocessed image
                output_image_path = os.path.join(output_path, image_name)
                cv2.imwrite(output_image_path, preprocessed_image)

print("Preprocessing completed for all images. Preprocessed images saved in", output_base_path)



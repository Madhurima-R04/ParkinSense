import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# Define the CNN model
def create_cnn(input_shape=(128, 128, 1)):
    cnn_model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Dropout(0.2),  # Add dropout to prevent overfitting
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Dropout(0.3),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Dropout(0.4),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),  # Add dropout here as well
        Dense(1, activation='sigmoid')  # Binary classification
    ])
    cnn_model.compile(optimizer=Adam(learning_rate=0.0001),
                      loss='binary_crossentropy',
                      metrics=['accuracy'])
    return cnn_model

# Load preprocessed images using image_dataset_from_directory
train_dataset = tf.keras.utils.image_dataset_from_directory(
    'archive/drawings/spiral/training',  # Update this to match your directory structure
    labels='inferred',                   # Automatically infers labels based on folder names
    label_mode='binary',                 # For binary classification
    color_mode='grayscale',              # Use grayscale images
    batch_size=16,                       # Batch size
    image_size=(128, 128),               # Resize all images to 128x128
    shuffle=True                         # Shuffle the dataset
)

val_dataset = tf.keras.utils.image_dataset_from_directory(
    'archive/drawings/spiral/testing',   # Use testing data as validation during training
    labels='inferred',                   # Automatically infers labels based on folder names
    label_mode='binary',                 # For binary classification
    color_mode='grayscale',              # Use grayscale images
    batch_size=16,                       # Batch size
    image_size=(128, 128)                # Resize all images to 128x128
)

# Normalize the datasets
normalization_layer = tf.keras.layers.Rescaling(1./255)
train_dataset = train_dataset.map(lambda x, y: (normalization_layer(x), y))
val_dataset = val_dataset.map(lambda x, y: (normalization_layer(x), y))

# Optional: Prefetch data for faster training
train_dataset = train_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
val_dataset = val_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

# Instantiate and train the model
model = create_cnn()

# Add early stopping to prevent overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the model
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=30,  # Increase epochs for better learning
    callbacks=[early_stopping]
)

# Print the final training and validation accuracy
final_train_accuracy = history.history['accuracy'][-1]
final_val_accuracy = history.history['val_accuracy'][-1]
print(f"Final Training Accuracy: {final_train_accuracy * 100:.2f}%")
print(f"Final Validation Accuracy: {final_val_accuracy * 100:.2f}%")

# Save the trained model
model.save("cnn_model.h5")  # Saves the model as 'cnn_model.h5'
print("Model saved as cnn_model.h5")

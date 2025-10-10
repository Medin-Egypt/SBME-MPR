import tensorflow as tf
import numpy as np
import os

# --- Configuration ---
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
DATA_DIR = 'data'
NUM_CLASSES = 3  # Make sure this matches the number of folders in DATA_DIR
EPOCHS = 10      # Start with 10 and increase if needed

# --- 1. Load Data ---
# Create datasets from the directory structure
# It will automatically infer class names from folder names
# We split the data into 80% for training and 20% for validation
print("Loading and preparing datasets...")
train_dataset = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='int' # Use 'int' for SparseCategoricalCrossentropy
)

validation_dataset = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='int'
)

# Store the class names for later use in prediction
class_names = train_dataset.class_names
print(f"Found classes: {class_names}")

# Configure dataset for performance
AUTOTUNE = tf.data.AUTOTUNE
train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)


# --- 2. Build the Model (Transfer Learning) ---
print("Building model...")
# Load MobileNetV2, a powerful and lightweight pre-trained model
# include_top=False removes the original classification layer
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(*IMAGE_SIZE, 3),
    include_top=False,
    weights='imagenet'
)

# Freeze the base model's layers so we don't alter its learned weights
base_model.trainable = False

# Create our new model on top of the base model
inputs = tf.keras.Input(shape=(*IMAGE_SIZE, 3))
# The mobilenet_v2.preprocess_input function is a requirement for this model
x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
x = base_model(x, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.2)(x)  # Regularization
outputs = tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')(x) # Output layer for our 3 classes

model = tf.keras.Model(inputs, outputs)

# --- 3. Compile the Model ---
print("Compiling model...")
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)

model.summary()

# --- 4. Train the Model ---
print("\nStarting training...")
history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=EPOCHS
)
print("Training finished.")


# --- 5. Save the Model ---
# The modern .keras format is recommended
model.save('my_image_classifier.keras')
print(f"\nModel saved successfully as 'my_image_classifier.keras'")
# Also save class names for easy lookup during prediction
with open('class_names.txt', 'w') as f:
    for item in class_names:
        f.write("%s\n" % item)
print("Class names saved to 'class_names.txt'")
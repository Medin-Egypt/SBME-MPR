import tensorflow as tf
import numpy as np
import pydicom  # Used for reading DICOM files

# --- Configuration ---
MODEL_PATH = 'model/model.keras'
CLASS_NAMES_PATH = 'model/class_names.txt'
IMAGE_SIZE = (224, 224)

# --- Load the saved model and class names ---
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(CLASS_NAMES_PATH, 'r') as f:
        class_names = [line.strip() for line in f]
except IOError as e:
    print(f"Error loading model or class names file: {e}")


# --- Function to predict a single image ---
def predict_image(image_path):
    """Loads an image, preprocesses it, and returns the predicted class and confidence."""
    img = tf.keras.utils.load_img(image_path, target_size=(224, 224))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Create a batch

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    predicted_class = class_names[np.argmax(score)]
    confidence = 100 * np.max(score)

    return predicted_class, confidence


# --- Function to predict a single DICOM image ---
def predict_dicom_image(pixel_array):
    """Loads a DICOM image, preprocesses it, and returns the predicted class and confidence."""

    # 2. Normalize pixel values to the 0-255 range
    if pixel_array.dtype != np.uint8:
        pixel_array = pixel_array.astype(float)
        # Ensure array has non-zero max to avoid division by zero
        if pixel_array.max() > 0:
            pixel_array = (np.maximum(pixel_array, 0) / pixel_array.max()) * 255.0
        pixel_array = pixel_array.astype(np.uint8)

    # 3. Convert grayscale to 3-channel RGB by duplicating the channel
    if len(pixel_array.shape) == 2:  # Check if it's a 2D (grayscale) image
        img_rgb = np.stack([pixel_array] * 3, axis=-1)
    else:
        img_rgb = pixel_array  # Assume it's already in a compatible format

    # 4. Resize the image to what the model expects
    img_resized = tf.image.resize(img_rgb, IMAGE_SIZE)

    # 5. Create a batch and make a prediction
    img_array = tf.keras.utils.img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)  # Create a batch of 1

    predictions = model.predict(img_array, verbose=0)  # verbose=0 hides the progress bar
    score = tf.nn.softmax(predictions[0])

    predicted_class = class_names[np.argmax(score)]
    confidence = 100 * np.max(score)

    return predicted_class, confidence


if __name__ == '__main__':
    MODEL_PATH = '../model/model.keras'
    CLASS_NAMES_PATH = '../model/class_names.txt'
    try:
        ds = pydicom.dcmread(r"F:\CUFE-MPR\frontal\image-00001.dcm")
        pixel_array = ds.pixel_array
        predicted_class, confidence = predict_dicom_image(pixel_array)
        print(f"Prediction: This image is most likely a(n) {predicted_class} with {confidence:.2f}% confidence.")
    except Exception as e:
        print(f"An error occurred during prediction: {e}")

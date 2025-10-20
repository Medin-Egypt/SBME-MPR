import tensorflow as tf
import numpy as np
import os
import pydicom
from pydicom.errors import InvalidDicomError
from glob import glob

# Removed: from PIL import Image

# --- Configuration (Must match training script) ---
IMG_SIZE = 224
MODEL_PATH = 'model/model_final.keras'
if __name__ == '__main__':
    MODEL_PATH = '../model/model_final.keras'

# Define the class labels in the same order as your training data
# Update these labels based on your training script's 'Class mapping' output.
CLASS_LABELS = ['Axial', 'Coronal', 'Sagittal']

# Removed: OUTPUT_IMAGE_PATH = 'extracted_dicom_slice.png'

# Global variable for the loaded model
_model = None


def get_model():
    """Loads the model once and returns it for subsequent calls."""
    global _model
    if _model is None:
        try:
            _model = tf.keras.models.load_model(MODEL_PATH)
        except Exception as e:
            raise RuntimeError(f"Could not load the model from {MODEL_PATH}: {e}")
    return _model


def apply_dicom_windowing(pixel_array, ds):
    """
    Applies Rescale Slope/Intercept and Windowing (Center/Width) to the DICOM pixel data.
    """
    processed_array = pixel_array.astype(np.float32)

    # 1. Apply Rescale Slope and Intercept (Hounsfield Unit conversion)
    if 'RescaleSlope' in ds and 'RescaleIntercept' in ds:
        slope = ds.RescaleSlope
        intercept = ds.RescaleIntercept
        processed_array = processed_array * slope + intercept

    # 2. Apply Window Center (WC) and Window Width (WW) for display
    if 'WindowCenter' in ds and 'WindowWidth' in ds:
        wc = ds.WindowCenter
        ww = ds.WindowWidth

        # Handle multiple window settings if they exist (use the first one)
        if isinstance(wc, pydicom.multival.MultiValue):
            wc = wc[0]
        if isinstance(ww, pydicom.multival.MultiValue):
            ww = ww[0]

        # Calculate intensity bounds
        min_val = wc - (ww / 2)
        max_val = wc + (ww / 2)

        # Clip array to the window range
        processed_array = np.clip(processed_array, min_val, max_val)

        print(f"   -> Applied DICOM Windowing: WC={wc}, WW={ww}")

        # Normalize the windowed data to 0-255 range for display
        processed_array = (processed_array - min_val) / (max_val - min_val) * 255.0

    else:
        # Fallback: Simple min-max normalization to 0-255 if no windowing metadata exists
        print("   -> Windowing metadata (WC/WW) not found. Using min-max normalization.")
        if processed_array.max() > processed_array.min():
            processed_array = (processed_array - processed_array.min()) / (
                    processed_array.max() - processed_array.min()) * 255.0
        else:
            processed_array = np.zeros_like(processed_array)

    # Convert to 8-bit integer for model input preprocessing
    return processed_array.astype(np.uint8)


def preprocess_array(pixel_array_8bit):
    """
    Resizes the 8-bit pixel array and creates the model input array.

    NOTE: The return is simplified as the image-to-save is no longer needed.

    Args:
        pixel_array_8bit (np.ndarray): The 8-bit (0-255) processed image data.

    Returns:
        np.ndarray: The preprocessed array for model input (shape (1, IMG_SIZE, IMG_SIZE, 3), range [0, 1]).
    """
    img_array = pixel_array_8bit.astype(np.float32)

    # Convert grayscale (H, W) or (H, W, 1) to RGB (H, W, 3)
    if img_array.ndim == 2:
        img_array = np.stack([img_array] * 3, axis=-1)
    elif img_array.shape[-1] == 1:
        img_array = np.stack([img_array[:, :, 0]] * 3, axis=-1)

    # Use TensorFlow to resize the image
    img_tensor = tf.convert_to_tensor(img_array)
    resized_tensor = tf.image.resize(img_tensor, [IMG_SIZE, IMG_SIZE])

    # Convert to 8-bit integer for consistent data type before normalization
    image_for_model = tf.cast(resized_tensor, tf.uint8).numpy()

    # Add batch dimension and rescale (must match training 1./255) for model input
    processed_array_model_input = np.expand_dims(image_for_model.astype(np.float32), axis=0) / 255.0

    # Simplified return
    return processed_array_model_input


def run_prediction_on_preprocessed_array(input_array):
    """Performs the prediction."""
    model = get_model()
    predictions = model.predict(input_array, verbose=0)

    probabilities = predictions[0]
    predicted_index = np.argmax(probabilities)
    confidence = probabilities[predicted_index]
    predicted_label = CLASS_LABELS[predicted_index]

    all_probs = {label: prob for label, prob in zip(CLASS_LABELS, probabilities)}

    return predicted_label, float(confidence), all_probs


# ----------------------------------------------------------------------
# Updated DICOM Folder Function
# ----------------------------------------------------------------------

def predict_middle_dicom_from_folder(dicom_folder_path):
    """
    Locates the middle slice, applies DICOM scaling/windowing, and returns
    the prediction results.
    """
    if not os.path.isdir(dicom_folder_path):
        raise FileNotFoundError(f"Folder not found at: {dicom_folder_path}")

    dicom_files = glob(os.path.join(dicom_folder_path, '*'))

    # 1. Read DICOM metadata and filter
    slices = []
    for filepath in dicom_files:
        try:
            ds = pydicom.dcmread(filepath, force=True)
            if 'InstanceNumber' in ds and hasattr(ds, 'pixel_array'):
                slices.append(ds)
        except InvalidDicomError:
            continue

    if not slices:
        raise ValueError(f"No valid DICOM files with pixel data found in {dicom_folder_path}.")

    # 2. Sort the slices by Instance Number
    slices.sort(key=lambda x: int(x.InstanceNumber))

    # 3. Locate the middle slice
    middle_index = len(slices) // 2
    middle_slice = slices[middle_index]

    print(
        f"Found {len(slices)} DICOM slices. Predicting on slice {middle_index + 1} (Instance Number: {middle_slice.InstanceNumber}).")

    # 4. Apply DICOM windowing and scaling to get 8-bit image data
    pixel_array_8bit = apply_dicom_windowing(middle_slice.pixel_array, middle_slice)

    # 5. Preprocess for model
    # The return of preprocess_array is now a single array
    input_image = preprocess_array(pixel_array_8bit)

    # 6. Predict
    return run_prediction_on_preprocessed_array(input_image)


# ----------------------------------------------------------------------
# Example Usage (Main block)
# ----------------------------------------------------------------------

if __name__ == "__main__":
    # NOTE: You MUST replace this path with the path to a folder
    # containing multiple DICOM files that form a series.
    TEST_DICOM_FOLDER = r"F:\CUFE-MPR\full"

    print("\n--- Testing predict_middle_dicom_from_folder ---")
    try:
        predicted_class, confidence, all_probs = predict_middle_dicom_from_folder(TEST_DICOM_FOLDER)
        print(f"\nFolder Path: {TEST_DICOM_FOLDER}")
        print(f"Inference Image Size: {IMG_SIZE}x{IMG_SIZE} pixels")
        print(f"Predicted Class: {predicted_class}")
        print(f"Confidence: {confidence:.4f}")
        print(f"All Probabilities: {all_probs}")
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        print(f"\nError running DICOM folder test: {e}")
        print(
            "Please ensure 'pydicom' is installed, the MODEL_PATH is correct, and the TEST_DICOM_FOLDER path points to valid DICOM files.")
"""
Traffic Sign Classification - Model & Preprocessing Utilities.

Provides reusable components for:
- Building and loading the EfficientNetB0 transfer learning model
- Loading class index to traffic sign name mappings
- Validating and preprocessing input images
- Generating top-k probabilistic predictions
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, Union

import numpy as np
import joblib
from PIL import Image, UnidentifiedImageError
import tensorflow as tf


DEFAULT_MODEL_KERAS_PATH = Path("models/traffic_sign_model.keras")
DEFAULT_WEIGHTS_PATH = Path("models/model_weights.pkl")
ALT_WEIGHTS_PATH = Path("model_weights.pkl")
DEFAULT_MAPPING_PATH = Path("models/class_mapping.json")
ALT_MAPPING_PATH = Path("class_mapping.json")

TARGET_IMAGE_SIZE = (128, 128)
INPUT_SHAPE = (128, 128, 3)
EXPECTED_NUM_CLASSES = 49


def create_model_architecture(
    input_shape: Tuple[int, int, int] = INPUT_SHAPE,
    num_classes: int = EXPECTED_NUM_CLASSES
) -> tf.keras.Model:
    """
    Reconstructs the exact transfer learning model architecture used in training:
    - EfficientNetB0 backbone (include_top=False, weights=None, pooling='avg')
    - BatchNormalization layer
    - Dense layer with 512 units and ReLU activation
    - Softmax output Dense layer with num_classes units
    """
    inputs = tf.keras.layers.Input(shape=input_shape, name="input_image")
    base_model = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights=None,
        pooling="avg"
    )
    base_model.trainable = False

    x = base_model(inputs)
    x = tf.keras.layers.BatchNormalization(name="batch_normalization")(x)
    x = tf.keras.layers.Dense(512, activation="relu", name="dense_512")(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="output_classes")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="traffic_sign_classifier")
    return model


def load_class_mapping(mapping_path: Optional[Union[str, Path]] = None) -> Dict[int, Dict[str, Any]]:
    """
    Loads and validates the class index mapping from JSON.
    Maps model output index (0..48) to sign metadata (name, folder, category, action).
    """
    search_paths = []
    if mapping_path:
        search_paths.append(Path(mapping_path))
    search_paths.extend([DEFAULT_MAPPING_PATH, ALT_MAPPING_PATH])

    resolved_path = None
    for p in search_paths:
        if p.exists():
            resolved_path = p
            break

    if resolved_path is None:
        raise FileNotFoundError(
            f"Class mapping file not found. Searched: {[str(p) for p in search_paths]}. "
            "Please ensure 'models/class_mapping.json' exists in the project root."
        )

    with open(resolved_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Convert string keys to integer keys
    mapping = {int(k): v for k, v in data.items()}

    if len(mapping) != EXPECTED_NUM_CLASSES:
        raise ValueError(
            f"Expected {EXPECTED_NUM_CLASSES} classes in mapping, but found {len(mapping)}."
        )

    return mapping


def load_model(
    model_path: Optional[Union[str, Path]] = None,
    weights_path: Optional[Union[str, Path]] = None,
    num_classes: int = EXPECTED_NUM_CLASSES
) -> Tuple[tf.keras.Model, str]:
    """
    Loads the trained model using the best available artifact:
    1. Direct native Keras format (e.g. models/traffic_sign_model.keras)
    2. Reconstructed architecture populated with weights from model_weights.pkl

    Returns:
        Tuple of (model, source_description)
    """
    # 1. Try loading direct .keras model if path specified or exists at default
    candidate_keras_paths = []
    if model_path:
        candidate_keras_paths.append(Path(model_path))
    candidate_keras_paths.append(DEFAULT_MODEL_KERAS_PATH)

    for p in candidate_keras_paths:
        if p.exists() and p.is_file():
            try:
                model = tf.keras.models.load_model(p, compile=False)
                # Verify output layer size
                output_units = model.output_shape[-1]
                if output_units != num_classes:
                    raise ValueError(
                        f"Model output size {output_units} does not match expected {num_classes} classes."
                    )
                return model, f"Native Keras model loaded from {p}"
            except Exception as e:
                # If loading fails, log and continue to weights reconstruction
                print(f"Warning: Could not load {p} directly ({e}). Attempting weights reconstruction...")

    # 2. Try loading weights into reconstructed architecture
    candidate_weight_paths = []
    if weights_path:
        candidate_weight_paths.append(Path(weights_path))
    candidate_weight_paths.extend([DEFAULT_WEIGHTS_PATH, ALT_WEIGHTS_PATH])

    resolved_weights_path = None
    for p in candidate_weight_paths:
        if p.exists() and p.is_file():
            resolved_weights_path = p
            break

    if resolved_weights_path is None:
        raise FileNotFoundError(
            "No valid model artifacts found! Please ensure either "
            f"'{DEFAULT_MODEL_KERAS_PATH}' or '{DEFAULT_WEIGHTS_PATH}' exists. "
            "Refer to the README for model weight setup instructions."
        )

    try:
        model = create_model_architecture(input_shape=INPUT_SHAPE, num_classes=num_classes)
        weights = joblib.load(resolved_weights_path)
        model.set_weights(weights)
        return model, f"Reconstructed architecture with weights from {resolved_weights_path}"
    except Exception as e:
        raise RuntimeError(f"Failed to load weights from {resolved_weights_path}: {e}") from e


def preprocess_image(
    image_input: Union[str, Path, bytes, Image.Image, Any],
    target_size: Tuple[int, int] = TARGET_IMAGE_SIZE
) -> Tuple[np.ndarray, Image.Image]:
    """
    Validates, standardizes, and preprocesses an input image:
    1. Loads or accepts PIL Image
    2. Converts to RGB (handles RGBA, grayscale, palette, etc.)
    3. Resizes to target_size (128, 128)
    4. Converts to numpy array of shape (1, 128, 128, 3)
    5. Applies tf.keras.applications.efficientnet.preprocess_input
    
    Returns:
        Tuple of (preprocessed_batch_tensor, rgb_pil_image)
    """
    if isinstance(image_input, Image.Image):
        pil_img = image_input
    else:
        try:
            pil_img = Image.open(image_input)
            pil_img.load()  # Force loading to verify image integrity
        except (UnidentifiedImageError, OSError, ValueError) as e:
            raise ValueError(f"Invalid or corrupted image file: {e}") from e

    # Ensure RGB conversion
    if pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")

    # Resize to model input dimensions (128x128)
    resized_img = pil_img.resize(target_size, Image.Resampling.BILINEAR)

    # Convert to array: shape (128, 128, 3), float32
    img_array = tf.keras.preprocessing.image.img_to_array(resized_img)

    # Validate shape
    if img_array.shape != (*target_size, 3):
        raise ValueError(f"Unexpected image dimensions: {img_array.shape}, expected {(*target_size, 3)}")

    # Add batch dimension: shape (1, 128, 128, 3)
    batch_array = np.expand_dims(img_array, axis=0)

    # Apply EfficientNet preprocessing exactly as during training
    preprocessed_tensor = tf.keras.applications.efficientnet.preprocess_input(batch_array)

    return preprocessed_tensor, pil_img


def predict(
    model: tf.keras.Model,
    preprocessed_tensor: np.ndarray,
    class_mapping: Dict[int, Dict[str, Any]],
    top_k: int = 3
) -> Dict[str, Any]:
    """
    Runs inference on the preprocessed tensor and maps results to class names.
    
    Returns structured results:
        - top_index: int
        - class_name: str
        - confidence: float (0.0 to 100.0)
        - category: str
        - action: str
        - top_k: list of dicts with top predictions
        - raw_probabilities: full 1D array of probabilities
    """
    if preprocessed_tensor.shape != (1, *INPUT_SHAPE):
        raise ValueError(
            f"Input tensor shape {preprocessed_tensor.shape} does not match expected (1, {INPUT_SHAPE[0]}, {INPUT_SHAPE[1]}, {INPUT_SHAPE[2]})."
        )

    # Run inference
    preds = model.predict(preprocessed_tensor, verbose=0)
    probabilities = preds[0]

    num_classes = len(class_mapping)
    if len(probabilities) != num_classes:
        raise ValueError(
            f"Prediction output dimension ({len(probabilities)}) does not match class mapping count ({num_classes})."
        )

    # Sort descending
    sorted_indices = np.argsort(probabilities)[::-1]
    top_index = int(sorted_indices[0])
    top_confidence = float(probabilities[top_index] * 100.0)
    top_info = class_mapping.get(top_index, {
        "name": f"Class {top_index}",
        "category": "Unknown",
        "action": "Proceed with caution."
    })

    top_k_list = []
    k_to_take = min(top_k, num_classes)
    for i in range(k_to_take):
        idx = int(sorted_indices[i])
        info = class_mapping.get(idx, {"name": f"Class {idx}", "category": "Unknown"})
        top_k_list.append({
            "index": idx,
            "class_name": info["name"],
            "confidence": float(probabilities[idx] * 100.0),
            "category": info.get("category", "General"),
            "action": info.get("action", "")
        })

    return {
        "top_index": top_index,
        "class_name": top_info["name"],
        "confidence": top_confidence,
        "category": top_info.get("category", "General"),
        "action": top_info.get("action", "Follow standard traffic guidelines."),
        "top_k": top_k_list,
        "raw_probabilities": probabilities
    }

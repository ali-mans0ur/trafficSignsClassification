"""
Verification Script for Traffic Sign Classification Pipeline.

Runs automated tests for:
1. Class mapping integrity and count
2. Model reconstruction and weights loading
3. Native Keras model loading
4. Preprocessing pipeline verification
5. End-to-end inference accuracy across sample test images
6. Robustness against corrupt or invalid input
"""

import sys
import io
from pathlib import Path
import numpy as np
from PIL import Image

import model_utils


def test_class_mapping():
    print("[1/5] Testing class mapping loading...")
    mapping = model_utils.load_class_mapping()
    assert len(mapping) == 49, f"Expected 49 classes, got {len(mapping)}"
    assert 0 in mapping, "Class index 0 missing"
    assert 48 in mapping, "Class index 48 missing"
    assert mapping[0]["name"] == "Speed limit (5km/h)", f"Index 0 name mismatch: {mapping[0]['name']}"
    assert mapping[2]["name"] == "Dont Go straight", f"Index 2 name mismatch: {mapping[2]['name']}"
    assert mapping[10]["name"] == "Speed limit (30km/h)", f"Index 10 name mismatch: {mapping[10]['name']}"
    print("  -> Class mapping verified: 49 classes correctly mapped.")
    return mapping


def test_model_loading():
    print("[2/5] Testing model loading...")
    # Test loading
    model, source = model_utils.load_model()
    print(f"  -> Model loaded successfully: {source}")
    assert model.input_shape == (None, 128, 128, 3), f"Unexpected input shape: {model.input_shape}"
    assert model.output_shape == (None, 49), f"Unexpected output shape: {model.output_shape}"
    return model


def test_preprocessing():
    print("[3/5] Testing image preprocessing pipeline...")
    # Create synthetic test image
    test_img = Image.new("RGBA", (256, 256), color=(255, 0, 0, 128))
    tensor, pil_rgb = model_utils.preprocess_image(test_img)
    assert tensor.shape == (1, 128, 128, 3), f"Wrong tensor shape: {tensor.shape}"
    assert pil_rgb.mode == "RGB", f"Mode is not RGB: {pil_rgb.mode}"
    print("  -> Preprocessing passed: RGBA to RGB conversion & 128x128 batch shape verified.")

    # Test corrupted image handling
    try:
        model_utils.preprocess_image(io.BytesIO(b"not an image file"))
        assert False, "Should have raised ValueError on corrupt image"
    except ValueError:
        print("  -> Corrupted image error handling verified.")


def test_inference_on_samples(model, mapping):
    print("[4/5] Testing end-to-end inference on real sample images...")
    sample_dir = Path("assets/samples")
    if not sample_dir.exists():
        print("  -> No assets/samples directory found, skipping sample test.")
        return

    expected_labels = {
        "speed_limit_5.png": "Speed limit (5km/h)",
        "dont_go_straight.png": "Dont Go straight",
        "dont_overtake_left.png": "Dont overtake from Left",
        "go_right.png": "Go Right",
        "uturn.png": "Uturn",
        "danger_ahead.png": "Danger Ahead",
        "dangerous_curve_left.png": "Dangerous curve to the left",
        "no_entry.png": "No entry"
    }

    correct = 0
    total = 0
    for file_name, expected_name in expected_labels.items():
        img_path = sample_dir / file_name
        if not img_path.exists():
            continue
        total += 1
        tensor, _ = model_utils.preprocess_image(img_path)
        result = model_utils.predict(model, tensor, mapping, top_k=3)
        top_name = result["class_name"]
        confidence = result["confidence"]
        is_match = top_name.lower() == expected_name.lower()
        if is_match:
            correct += 1
        status = "PASS" if is_match else "MISMATCH"
        print(f"  [{status}] {file_name}: Predicted '{top_name}' ({confidence:.1f}%) | Expected: '{expected_name}'")

    print(f"  -> Sample Accuracy: {correct}/{total} ({correct/total*100:.1f}%)")
    assert correct >= total - 1, f"Accuracy too low: {correct}/{total}"


def test_top_k_structure(model, mapping):
    print("[5/5] Testing Top-K prediction formatting...")
    dummy_input = np.zeros((1, 128, 128, 3), dtype=np.float32)
    res = model_utils.predict(model, dummy_input, mapping, top_k=3)
    assert "top_index" in res
    assert "class_name" in res
    assert "confidence" in res
    assert "top_k" in res
    assert len(res["top_k"]) == 3, f"Expected 3 top predictions, got {len(res['top_k'])}"
    print("  -> Top-K result structure verified.")


if __name__ == "__main__":
    print("==================================================")
    print("Running Traffic Sign Classification Verification")
    print("==================================================")
    try:
        mapping = test_class_mapping()
        model = test_model_loading()
        test_preprocessing()
        test_inference_on_samples(model, mapping)
        test_top_k_structure(model, mapping)
        print("==================================================")
        print("ALL PIPELINE TESTS PASSED SUCCESSFULLY! (5/5)")
        print("==================================================")
    except Exception as e:
        print(f"\nFAILED: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)

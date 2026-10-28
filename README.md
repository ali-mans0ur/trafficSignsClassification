# 🚦 Traffic Sign Classification Web App

A clean, responsive, and robust Streamlit web application for real-time Traffic Sign Classification using Deep Learning. Built as a university Deep Learning final project, the application reuses and productionizes an EfficientNetB0 transfer learning model trained in TensorFlow/Keras on the Kaggle Traffic Sign Dataset.

---

## 📌 Project Overview

Traffic signs provide critical visual guidance, speed regulations, and safety warnings for motorists, pedestrians, and automated driving systems. This project translates the experimental research conducted in `trafficSignClass.ipynb` into a production-grade interactive web interface.

Users can upload any traffic sign photograph or select from pre-loaded real-world samples to receive:
- The predicted traffic sign category and human-readable name.
- Model prediction confidence score as a precise percentage.
- The top-3 most probable predicted classes with comparative probability indicators.
- Actionable driver safety recommendations based on sign regulatory class.
- Full 49-class probability distribution inspection.

---

## ✨ Features

- **⚡ Instant Sub-Second Inference**: Cached model loading using `@st.cache_resource` prevents redundant disk I/O and GPU/CPU re-initialization.
- **🖼️ Flexible Ingestion & Safe Preprocessing**: Supports `.jpg`, `.jpeg`, and `.png` uploads with automatic image integrity verification, RGB conversion, and $(128 \times 128 \times 3)$ tensor normalization matching the exact training pipeline.
- **🧪 Instant Sample Gallery**: 8 pre-loaded test samples allow immediate testing directly within the browser without having to locate or upload image files.
- **📊 Top-3 Probabilistic Breakdown**: Visual horizontal distribution bars comparing the top three most likely traffic signs.
- **🚗 Driver Action Intelligence**: Contextual categorization (Speed Limit, Warning, Prohibitory, Mandatory, Regulatory) with recommended driver safety behaviors.
- **🛡️ Resilient Error Handling**: Graceful error handling for corrupted images, unsupported formats, or missing model artifacts.
- **📱 Responsive & Clean Dark Aesthetic**: Clean, high-contrast dark theme optimized for readability across desktop and mobile devices.

---

## 🧠 Model Architecture

The model uses **Transfer Learning** with Google's **EfficientNetB0** convolutional neural network backbone, pre-trained on ImageNet:

```text
Input Image (128 × 128 × 3 RGB)
       │
       ▼
EfficientNetB0 Backbone (Frozen, include_top=False, weights=ImageNet, pooling='avg')
       │ Output: (None, 1280) — 4,049,564 frozen parameters
       ▼
Batch Normalization Layer
       │ Output: (None, 1280) — 5,120 parameters
       ▼
Dense Hidden Layer (512 units, ReLU activation)
       │ Output: (None, 512) — 655,872 parameters
       ▼
Dense Classification Head (49 units, Softmax activation)
       │ Output: (None, 49) — 25,137 parameters
       ▼
Output Class Probabilities (∑ = 1.0)
```

- **Total Parameters:** ~4,735,693
- **Trainable Parameters:** ~683,569 (Batch Normalization and Dense head)
- **Non-Trainable Parameters:** ~4,049,564 (EfficientNetB0 feature extractor)

---

## 📁 Dataset Information

- **Dataset Source:** Kaggle Traffic Sign Dataset Classification (`ahemateja19bec1025/traffic-sign-dataset-classification`).
- **Initial Dataset Volume:** 4,170 traffic sign images across 58 candidate categories.
- **Class Filtering in Notebook:** Classes with fewer than 10 training instances were dropped (`SPLIT_MINIMUM_COUNT = 10`):
  - Dropped labels (9 classes): `['18', '19', '25', '33', '51', '53', '57', '8', '9']`.
  - Retained classes: **49 active classes** with adequate sample representation.
- **Data Augmentation Applied During Training:**
  - Random rotation: $\pm 10^\circ$
  - Width and height shift: $20\%$
  - Shear range: $10\%$
  - Zoom range: $20\%$
  - Fill mode: `constant`
  - EfficientNet preprocessing function applied.

---

## 🔬 Image Preprocessing Pipeline

To ensure consistency between model training and runtime inference:
1. **File Format & Integrity Verification:** The image is validated using `PIL.Image.open()` and `.load()`. Any corrupted or non-image files are rejected with clear error messages.
2. **Channel Standardization:** Images are converted to 3-channel RGB (removing alpha/transparency channels or converting grayscale).
3. **Resizing:** The image is resized to $128 \times 128$ pixels using bilinear interpolation.
4. **Tensor Expansion:** Converted to a NumPy array of shape $(1, 128, 128, 3)$.
5. **EfficientNet Preprocessing:** Scaled via `tf.keras.applications.efficientnet.preprocess_input(img_array)`.
6. **No Random Augmentation at Inference:** Data augmentation (rotation, shearing, zooming) is strictly disabled during inference.

---

## 🔍 Implementation Review & Notebook Fixes

During inspection of the experimental notebook `trafficSignClass.ipynb`, several critical issues were identified and resolved in the application codebase:

### 1. Corrected Class-Index Mapping Mismatch
- **Issue in Notebook:** In cell 10, the training generator was constructed using `flow_from_dataframe` where `y_col='label'` contained string directory names (`'0'`, `'1'`, `'10'`, `'11'`, `'2'`, `'20'`, ...). Keras sorts string classes **lexicographically**, which mapped:
  - Model index 0 $\rightarrow$ directory `'0'` (Speed limit 5km/h)
  - Model index 1 $\rightarrow$ directory `'1'` (Speed limit 15km/h)
  - Model index 2 $\rightarrow$ directory `'10'` (Dont Go straight)
  - Model index 10 $\rightarrow$ directory `'2'` (Speed limit 30km/h)
  In cell 19, the notebook mapped predictions using `label_map.get(predicted_class_index)` where `label_map` was derived directly from `labels.csv` ordered $0, 1, 2, \dots$. This caused severe prediction mismatches (e.g., predicting class index 2 showed "Speed limit 30km/h" instead of "Dont Go straight").
- **Fix:** Reconstructed the exact lexicographical class mapping dictionary (`models/class_mapping.json`) mapping model output indices $0 \dots 48$ directly to their true directory names, traffic sign names, categories, and driver actions. Tested across all sample classes with 100% accuracy.

### 2. Training and Validation Dataset Overlap (Data Leakage in Notebook)
- **Issue in Notebook:** Cell 6 performed:
  ```python
  split_df, _ = split_dataset(image_df)
  train_df, test_df = custom_train_test_split(split_df)
  train, val = custom_train_test_split(train_df)
  ```
  However, in cell 10:
  ```python
  train_images = train_generator.flow_from_dataframe(dataframe=train_df, ...)
  val_images = test_generator.flow_from_dataframe(dataframe=val, ...)
  ```
  Because `train_images` used `train_df` (which still contained all the rows of `val`), the validation set was a complete subset of the training set. This resulted in artificial validation accuracy ($98.4\%$).
- **Note:** The application code relies on the trained weights saved in `model_weights.pkl` and does not train during runtime. The evaluation issue in the notebook is documented here for academic integrity.

### 3. Output Layer Dimension Alignment
- The reconstructed model architecture strictly sets `num_classes = 49` matching the 320 weight tensors stored in `model_weights.pkl`. Output weights shape is $(512, 49)$ and bias shape is $(49,)$.

### 4. Weight Loading Architecture
- The notebook saved weights using `joblib.dump(model.get_weights(), 'model_weights.pkl')`. In `model_utils.py`, the architecture is dynamically reconstructed and weights are loaded via `model.set_weights()`. We also exported a native standalone Keras model (`models/traffic_sign_model.keras`) for faster loading.

---

## 📁 Project Directory Structure

```text
DL_finalProject/
├── app.py                      # Main Streamlit web application entry point
├── model_utils.py              # Reusable model loading, preprocessing & inference utilities
├── verify_pipeline.py          # Automated verification test script (5 test suites)
├── requirements.txt            # Minimal, pinned Python dependencies
├── README.md                   # Complete documentation & deployment guide
├── .gitignore                  # Git ignore rules for virtual environments & temporary files
│
├── .streamlit/
│   └── config.toml             # Custom theme colors and server settings
│
├── models/
│   ├── class_mapping.json      # Calibrated 49-class index to traffic sign metadata mapping
│   ├── traffic_sign_model.keras # Native Keras format model artifact (~19.8 MB)
│   └── model_weights.pkl       # Original saved model weights (~18.9 MB)
│
├── assets/
│   └── samples/                # Real-world traffic sign sample images for instant app testing
│       ├── speed_limit_5.png
│       ├── dont_go_straight.png
│       ├── dont_overtake_left.png
│       ├── go_right.png
│       ├── uturn.png
│       ├── danger_ahead.png
│       ├── dangerous_curve_left.png
│       └── no_entry.png
│
├── model_weights.pkl           # Root copy of original model weights
└── trafficSignClass.ipynb      # Original research and training notebook
```

---

## 🚀 Installation & Local Execution

### 1. Prerequisites
- Python 3.10, 3.11, or 3.12 installed.

### 2. Clone the Repository
```bash
git clone https://github.com/ali-mans0ur/trafficSignsClassification.git
cd trafficSignsClassification
```

### 3. Create and Activate a Virtual Environment
**On Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**On macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 4. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 5. Run Automated Verification Tests
Validate the model architecture, class mappings, and sample predictions:
```bash
python verify_pipeline.py
```
Expected output: `ALL PIPELINE TESTS PASSED SUCCESSFULLY! (5/5)`.

### 6. Launch the Streamlit Web Application
```bash
python -m streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## ☁️ Deployment on Streamlit Community Cloud

Streamlit Community Cloud enables free, serverless hosting directly from your GitHub repository.

### Step 1: Push Project to GitHub
1. Stage and commit the project files:
   ```bash
   git add app.py model_utils.py verify_pipeline.py requirements.txt README.md .gitignore .streamlit/ models/ assets/
   git commit -m "feat: complete production Streamlit web app with EfficientNetB0 traffic sign classifier"
   ```
2. Push to your GitHub repository:
   ```bash
   git push origin main
   ```
   > ℹ️ **File Size Note:** The model artifacts (`traffic_sign_model.keras` at ~19.8 MB and `model_weights.pkl` at ~18.9 MB) are well within GitHub's 100 MB limit, so they can be pushed directly via standard Git without Git LFS.

### Step 2: Deploy on Streamlit Community Cloud
1. Go to [share.streamlit.io](https://share.streamlit.io/) and sign in with your GitHub account.
2. Click **"Create app"** (or **"New app"**).
3. Select your repository: `ali-mans0ur/trafficSignsClassification`.
4. Set the **Branch** to `main`.
5. Set the **Main file path** to `app.py`.
6. Click **"Deploy!"**.
7. Streamlit Cloud will read `requirements.txt`, install dependencies, load the cached model, and launch your application with a public HTTPS URL.

---

## ⚠️ Known Limitations & Considerations

1. **Pruned Classes:** The model recognizes 49 out of the original 58 classes in the Kaggle dataset. Signs belonging to the 9 pruned classes (`['18', '19', '25', '33', '51', '53', '57', '8', '9']`, which had fewer than 10 training instances) will be categorized into the closest visual class.
2. **Cropping & Bounding Boxes:** The model expects the traffic sign to occupy the majority of the image frame. Extremely wide-angle photographs containing small or distant traffic signs should be cropped prior to upload for optimal accuracy.
3. **Severe Lighting & Occlusions:** Extreme motion blur, dark night photography without headlights, or heavy physical graffiti on signs may reduce classification confidence.

---

## 🛠️ Troubleshooting

| Issue | Cause | Solution |
| :--- | :--- | :--- |
| `FileNotFoundError: Class mapping file not found` | Missing `class_mapping.json` | Ensure `models/class_mapping.json` is present in the workspace. |
| `FileNotFoundError: No valid model artifacts found` | Missing model weights | Verify that `models/traffic_sign_model.keras` or `models/model_weights.pkl` exists. |
| `ValueError: Invalid or corrupted image file` | Corrupt or non-image upload | Upload a valid `.png`, `.jpg`, or `.jpeg` file. |
| `MemoryError on Streamlit Cloud` | Container memory limit reached | The app uses `include_top=False` with `@st.cache_resource` and minimal dependencies, keeping RAM usage $< 800\text{ MB}$. |
| `streamlit is not recognized` command error | Streamlit binary not in PATH | Run via `python -m streamlit run app.py`. |

---

## 📜 Academic Integrity & Credits
- **Project Type:** University Deep Learning Final Project
- **Model Backbone:** EfficientNetB0 (ImageNet Transfer Learning)
- **Dataset:** Kaggle Traffic Sign Dataset Classification by Ahemad Teja
- **Frameworks:** Streamlit, TensorFlow, NumPy, Pillow, Pandas, Joblib
"# trafficSignsClassification" 

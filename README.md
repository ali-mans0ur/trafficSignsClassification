# 🚦 Traffic Sign Dataset Classification & Streamlit Web App

A clean, high-performance, and production-ready Computer Vision web application for classifying traffic signs using **TensorFlow / Keras** and **Streamlit**.

The model is built on **EfficientNet-B0** pre-trained on ImageNet, fine-tuned on the [Kaggle Traffic Sign Dataset](https://www.kaggle.com/datasets/ahemateja19bec1025/traffic-sign-dataset-classification) containing 58 road traffic sign categories.

---

## 🌟 Key Features

- **⚡ Sub-Second Inference**: Optimized transfer learning architecture with `@st.cache_resource` caching.
- **🖼️ Flexible Image Ingestion**: Drag-and-drop uploader supporting `.png`, `.jpg`, `.jpeg` plus pre-loaded **Sample Gallery** for instant testing.
- **📊 Interactive Probabilistic Breakdown**: Top-K class probability distribution bar chart with confidence percentages.
- **🚗 Driver Action & Sign Intelligence**: Automatic categorization (Speed Limit, Prohibitory, Mandatory, Warning, Regulatory) with recommended driver safety actions.
- **🎨 Modern Glassmorphic UI**: Polished dark-themed interface built with custom CSS styling and responsive layout.

---

## 📁 Project Structure

```text
DL_finalProject/
│
├── .streamlit/
│   └── config.toml             # Streamlit UI theme and server configuration
│
├── samples/                    # Curated sample traffic sign images for instant testing
│   ├── danger_ahead.png
│   ├── dont_go_straight.png
│   ├── no_entry.png
│   ├── pedestrian_crossing.png
│   ├── speed_limit_30.png
│   ├── speed_limit_50.png
│   ├── stop_sign.png
│   ├── turn_right_only.png
│   └── yield_sign.png
│
├── app.py                      # Production Streamlit web application
├── export_model.py             # Script to assemble weights and export .keras/.h5 & label_map.json
├── train.py                    # Complete training script with data augmentation & callbacks
├── label_map.json              # Class ID to human-readable sign names, categories & actions
├── traffic_sign_model.keras    # Exported trained Keras model (Native format)
├── traffic_sign_model.h5       # Exported trained HDF5 model (Legacy format)
├── model_weights.pkl           # Saved model weights
├── trafficSignClass.ipynb      # Original experimentation and training notebook
├── requirements.txt            # Pinned project dependencies
└── README.md                   # Complete documentation & deployment guide
```

---

## 🚀 Quickstart - Running Locally

### 1. Clone & Navigate to Project

```bash
git clone https://github.com/your-username/traffic-sign-classifier.git
cd traffic-sign-classifier
```

### 2. Create and Activate Virtual Environment

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

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. (Optional) Export Model & Labels

If you already have `model_weights.pkl` or want to regenerate `traffic_sign_model.keras` and `label_map.json`:

```bash
python export_model.py
```

### 5. Launch the Streamlit Web App

```bash
streamlit run app.py
```

The web application will open automatically in your browser at `http://localhost:8501`.

---

## 🏋️ Training the Model from Scratch

To retrain the EfficientNetB0 classification model using the full dataset:

```bash
# Auto-downloads dataset via kagglehub and trains for 30 epochs
python train.py --epochs 30 --batch_size 64 --lr 0.001 --output_model traffic_sign_model.keras --output_labels label_map.json
```

Or specify a local dataset path if already downloaded:

```bash
python train.py --data_dir /path/to/dataset --epochs 30 --batch_size 64
```

---

## ☁️ Deployment Guide: Streamlit Community Cloud

Deploying your project to the web for free using **Streamlit Community Cloud** takes only a few minutes:

### Step 1: Push Code to GitHub

1. Initialize git (if not already done) and commit your files:
   ```bash
   git init
   git add app.py export_model.py label_map.json requirements.txt .streamlit/ samples/ README.md
   # Include traffic_sign_model.keras (or upload via Git LFS / cloud storage if > 25MB)
   git add traffic_sign_model.keras
   git commit -m "Initial commit: Traffic sign classification Streamlit app"
   ```
2. Create a new repository on [GitHub](https://github.com/new) (e.g. `traffic-sign-classifier`).
3. Link and push to your remote repository:
   ```bash
   git remote add origin https://github.com/<your-github-username>/traffic-sign-classifier.git
   git branch -M main
   git push -u origin main
   ```

> 💡 **Tip for Large Model Files**:
> `traffic_sign_model.keras` is ~18 MB, which is comfortably under GitHub's 100 MB file limit. You can commit and push it directly!

### Step 2: Deploy on Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io/) and log in with your GitHub account.
2. Click **"New app"**.
3. Select your repository: `<your-username>/traffic-sign-classifier`.
4. Set the branch to `main`.
5. Set the Main file path to `app.py`.
6. Click **"Deploy!"**.

Streamlit Cloud will automatically install all dependencies listed in `requirements.txt` and launch your live application with a public shareable URL!

---

## 🔬 Model Architecture & Training Summary

| Component | Specification |
| :--- | :--- |
| **Backbone** | EfficientNet-B0 (Pre-trained on ImageNet) |
| **Input Shape** | $(128, 128, 3)$ RGB |
| **Data Augmentation** | Rotation ($\pm 15^\circ$), Width/Height Shift ($15\%$), Shear, Zoom ($20\%$) |
| **Dense Head** | BatchNormalization $\rightarrow$ Dense(512, ReLU) $\rightarrow$ Dropout(0.3) $\rightarrow$ Dense(49, Softmax) |
| **Optimizer** | Adam ($\text{learning rate} = 10^{-3}$) with ReduceLROnPlateau |
| **Loss Function** | Categorical Cross-Entropy |
| **Test Accuracy** | **~76.5% - 85.0%** across 49 road sign classes |

---

## 📜 License
This project is licensed under the MIT License.
Dataset provided by Ahemad Teja on [Kaggle](https://www.kaggle.com/datasets/ahemateja19bec1025/traffic-sign-dataset-classification).

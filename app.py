"""
Traffic Sign Classification Web Application.

A modern, responsive, and robust Streamlit web application for real-time
traffic sign recognition using a fine-tuned EfficientNetB0 deep learning model.
"""

from pathlib import Path
from typing import Optional, Dict, Any

import streamlit as st
from PIL import Image
import pandas as pd

import model_utils


# ---------------------------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Traffic Sign Classification",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------------------------
# Custom CSS for Professional Dark-Mode Aesthetics
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    /* Main container and font */
    .main {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    
    /* Header card */
    .hero-card {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.9) 0%, rgba(15, 23, 42, 0.95) 100%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 24px 28px;
        margin-bottom: 24px;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3);
    }
    .hero-title {
        font-size: 2.2rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        margin-bottom: 8px;
        color: #F8FAFC;
    }
    .hero-subtitle {
        font-size: 1.05rem;
        color: #94A3B8;
        line-height: 1.5;
        margin: 0;
    }

    /* Result Card */
    .result-card {
        background: linear-gradient(145deg, #1E293B, #0F172A);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 20px;
        margin-top: 15px;
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.25);
    }
    .result-name {
        font-size: 1.6rem;
        font-weight: 700;
        color: #38BDF8;
        margin-bottom: 4px;
    }
    .result-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        margin-bottom: 12px;
    }
    .badge-speed { background-color: rgba(239, 68, 68, 0.2); color: #F87171; border: 1px solid rgba(239, 68, 68, 0.4); }
    .badge-warning { background-color: rgba(245, 158, 11, 0.2); color: #FBBF24; border: 1px solid rgba(245, 158, 11, 0.4); }
    .badge-prohibitory { background-color: rgba(220, 38, 38, 0.25); color: #FCA5A5; border: 1px solid rgba(220, 38, 38, 0.5); }
    .badge-mandatory { background-color: rgba(59, 130, 246, 0.2); color: #60A5FA; border: 1px solid rgba(59, 130, 246, 0.4); }
    .badge-general { background-color: rgba(148, 163, 184, 0.2); color: #CBD5E1; border: 1px solid rgba(148, 163, 184, 0.4); }
    
    .action-box {
        background-color: rgba(30, 41, 59, 0.6);
        border-left: 4px solid #38BDF8;
        padding: 10px 14px;
        border-radius: 6px;
        margin-top: 12px;
        font-size: 0.92rem;
        color: #E2E8F0;
    }

    /* Probability rows */
    .prob-row {
        margin-bottom: 12px;
    }
    .prob-header {
        display: flex;
        justify-content: space-between;
        font-size: 0.95rem;
        font-weight: 600;
        margin-bottom: 4px;
    }
    .prob-bar-bg {
        background-color: #334155;
        border-radius: 6px;
        height: 10px;
        overflow: hidden;
    }
    .prob-bar-fill {
        background: linear-gradient(90deg, #2563EB, #38BDF8);
        height: 100%;
        border-radius: 6px;
    }

    /* Sample image card button */
    .stButton > button {
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.2s ease-in-out;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Cached Model & Mapping Loaders
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading traffic sign model artifacts...")
def load_cached_model():
    """Caches model loading to avoid redundant disk I/O and reconstruction."""
    return model_utils.load_model()


@st.cache_resource(show_spinner="Loading class index mapping...")
def load_cached_mapping():
    """Caches class mapping metadata."""
    return model_utils.load_class_mapping()


# ---------------------------------------------------------------------------
# State Initialization
# ---------------------------------------------------------------------------
if "current_image" not in st.session_state:
    st.session_state.current_image = None
if "image_source_name" not in st.session_state:
    st.session_state.image_source_name = ""
if "prediction_result" not in st.session_state:
    st.session_state.prediction_result = None


def reset_app_state():
    """Clears the currently loaded image and prediction results."""
    st.session_state.current_image = None
    st.session_state.image_source_name = ""
    st.session_state.prediction_result = None


# ---------------------------------------------------------------------------
# Hero Header Section
# ---------------------------------------------------------------------------
st.markdown("""
<div class="hero-card">
    <div class="hero-title">🚦 Traffic Sign Classification</div>
    <div class="hero-subtitle">
        Deep Learning Computer Vision System powered by <strong>Transfer Learning</strong> with 
        <strong>EfficientNetB0</strong> and <strong>TensorFlow</strong>. Detects and categorizes road traffic signs 
        across 49 distinct regulatory, warning, and directional categories.
    </div>
</div>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Sidebar: System Metadata & Controls
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Model & System")
    
    # Load model & mapping
    try:
        model, model_source = load_cached_model()
        mapping = load_cached_mapping()
        st.success("🟢 Model Ready", icon="✅")
        st.caption(f"**Source:** {model_source}")
        st.caption(f"**Classes:** {len(mapping)} traffic sign categories")
    except Exception as e:
        st.error(f"🔴 Model Error: {e}", icon="⚠️")
        model = None
        mapping = None

    st.markdown("---")
    st.subheader("📋 Specifications")
    st.markdown("""
    - **Backbone:** EfficientNetB0 (ImageNet weights)
    - **Dense Head:** BatchNorm + Dense(512, ReLU)
    - **Output Layer:** Softmax (49 classes)
    - **Input Shape:** (128, 128, 3) RGB
    - **Framework:** TensorFlow / Keras
    """)

    st.markdown("---")
    st.subheader("💡 Quick Tips")
    st.markdown("""
    - Use clear, centered traffic sign photos.
    - Supported formats: `.png`, `.jpg`, `.jpeg`.
    - Try the **Sample Gallery** below for instant test examples.
    """)

    st.markdown("---")
    if st.button("🔄 Reset Application", use_container_width=True):
        reset_app_state()
        st.rerun()


# ---------------------------------------------------------------------------
# Main Layout: Two Columns (Input & Prediction)
# ---------------------------------------------------------------------------
col_input, col_result = st.columns([1, 1], gap="large")

with col_input:
    st.subheader("📤 1. Provide an Image")

    # Sample Gallery for instant evaluation
    sample_dir = Path("assets/samples")
    if sample_dir.exists():
        sample_files = list(sample_dir.glob("*.png"))
        if sample_files:
            st.markdown("##### Or choose a pre-loaded sample:")
            sample_labels = {
                "speed_limit_5.png": "Speed Limit 5 km/h",
                "dont_go_straight.png": "Don't Go Straight",
                "dont_overtake_left.png": "No Overtaking",
                "go_right.png": "Go Right",
                "uturn.png": "U-Turn",
                "danger_ahead.png": "Danger Ahead",
                "dangerous_curve_left.png": "Dangerous Curve Left",
                "no_entry.png": "No Entry"
            }

            sample_cols = st.columns(4)
            for idx, file_path in enumerate(sample_files[:8]):
                btn_label = sample_labels.get(file_path.name, file_path.stem.replace("_", " ").title())
                with sample_cols[idx % 4]:
                    if st.button(btn_label, key=f"sample_{idx}", use_container_width=True):
                        try:
                            img = Image.open(file_path)
                            st.session_state.current_image = img
                            st.session_state.image_source_name = f"Sample: {btn_label}"
                            st.session_state.prediction_result = None
                            st.rerun()
                        except Exception as ex:
                            st.error(f"Error loading sample: {ex}")

    st.markdown("##### Upload your own file:")
    uploaded_file = st.file_uploader(
        "Upload a traffic sign image",
        type=["png", "jpg", "jpeg"],
        help="Upload JPG, JPEG, or PNG images.",
        label_visibility="collapsed"
    )

    if uploaded_file is not None:
        try:
            pil_image = Image.open(uploaded_file)
            pil_image.load()
            st.session_state.current_image = pil_image
            st.session_state.image_source_name = uploaded_file.name
        except Exception as e:
            st.error(f"Invalid or corrupted image file: {e}")
            st.session_state.current_image = None
            st.session_state.image_source_name = ""

    # Image Preview & Classification Button
    if st.session_state.current_image is not None:
        st.markdown("---")
        st.markdown(f"**Loaded Image:** `{st.session_state.image_source_name}`")
        img_w, img_h = st.session_state.current_image.size
        st.image(
            st.session_state.current_image,
            caption=f"Dimensions: {img_w} × {img_h} px | Mode: {st.session_state.current_image.mode}",
            use_container_width=False,
            width=260
        )

        col_btn1, col_btn2 = st.columns([2, 1])
        with col_btn1:
            classify_clicked = st.button("🚀 Classify Traffic Sign", type="primary", use_container_width=True)
        with col_btn2:
            if st.button("Clear Image", use_container_width=True):
                reset_app_state()
                st.rerun()

        if classify_clicked:
            if model is None or mapping is None:
                st.error("Model artifacts or class mapping are missing. Please verify system setup.")
            else:
                with st.spinner("Processing image and predicting class..."):
                    try:
                        # Preprocess image
                        tensor, _ = model_utils.preprocess_image(st.session_state.current_image)
                        
                        # Run inference
                        result = model_utils.predict(model, tensor, mapping, top_k=3)
                        st.session_state.prediction_result = result
                    except Exception as err:
                        st.error(f"Prediction failed: {err}")
    else:
        st.info("👆 Please upload an image or select a sample sign to get started.")


with col_result:
    st.subheader("🎯 2. Classification Results")

    if st.session_state.prediction_result is not None:
        res = st.session_state.prediction_result
        top_name = res["class_name"]
        top_conf = res["confidence"]
        top_cat = res["category"]
        top_action = res["action"]

        # Category Badge styling
        badge_class = "badge-general"
        if "Speed" in top_cat:
            badge_class = "badge-speed"
        elif "Warning" in top_cat:
            badge_class = "badge-warning"
        elif "Prohibitory" in top_cat:
            badge_class = "badge-prohibitory"
        elif "Mandatory" in top_cat:
            badge_class = "badge-mandatory"

        # Result card HTML
        st.markdown(f"""
        <div class="result-card">
            <span class="result-badge {badge_class}">{top_cat}</span>
            <div class="result-name">{top_name}</div>
            <div style="font-size: 1.15rem; color: #E2E8F0; margin-bottom: 8px;">
                Confidence: <strong style="color: #38BDF8; font-size: 1.3rem;">{top_conf:.2f}%</strong>
            </div>
            <div class="action-box">
                <strong>🚗 Recommended Driver Action:</strong><br>{top_action}
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### 📊 Top 3 Predictions")
        for item in res["top_k"]:
            c_name = item["class_name"]
            c_conf = item["confidence"]
            st.markdown(f"""
            <div class="prob-row">
                <div class="prob-header">
                    <span>{c_name}</span>
                    <span style="color: #38BDF8;">{c_conf:.2f}%</span>
                </div>
                <div class="prob-bar-bg">
                    <div class="prob-bar-fill" style="width: {min(100.0, max(1.0, c_conf))}%;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Full distribution expander
        with st.expander("🔍 View All 49 Class Probabilities"):
            all_data = []
            for idx, prob in enumerate(res["raw_probabilities"]):
                info = mapping.get(idx, {"name": f"Class {idx}", "category": "General"})
                all_data.append({
                    "Model Index": idx,
                    "Sign Name": info["name"],
                    "Category": info.get("category", "General"),
                    "Probability (%)": round(float(prob * 100.0), 3)
                })
            df_probs = pd.DataFrame(all_data).sort_values("Probability (%)", ascending=False).reset_index(drop=True)
            st.dataframe(df_probs, use_container_width=True)

    else:
        st.markdown("""
        <div style="border: 2px dashed #334155; border-radius: 12px; padding: 48px; text-align: center; color: #64748B;">
            <div style="font-size: 2.5rem; margin-bottom: 12px;">📊</div>
            <div style="font-size: 1.1rem; font-weight: 600; color: #94A3B8;">Awaiting Inference</div>
            <p style="font-size: 0.9rem; margin-top: 6px;">
                Select an image on the left and click <strong>"Classify Traffic Sign"</strong> to view model predictions and confidence scores.
            </p>
        </div>
        """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Deep Dive Tabs: How It Works, Architecture, Technologies
# ---------------------------------------------------------------------------
st.markdown("---")
st.subheader("📚 Project Documentation & Architecture")

tab_pipeline, tab_arch, tab_dataset, tab_tech = st.tabs([
    "🔍 How the Model Works",
    "🧠 Neural Network Architecture",
    "📁 Dataset & Preprocessing",
    "🛠️ Technologies Used"
])

with tab_pipeline:
    st.markdown("""
    ### 🔄 End-to-End Inference Pipeline
    
    1. **Image Ingestion & Validation:**
       The uploaded image is loaded into memory, verified for file integrity, and converted to a standardized 3-channel **RGB** format (discarding any alpha transparency or CMYK artifacts).
    2. **Dimensional Standardization:**
       The image is resized to **128 × 128 pixels** using bilinear interpolation, aligning with the input shape required by the convolutional backbone.
    3. **EfficientNet Preprocessing:**
       The tensor is normalized using `tf.keras.applications.efficientnet.preprocess_input`. Unlike simple $[0, 1]$ division, this matches the exact distribution expected by the pre-trained EfficientNet weights.
    4. **Feature Extraction:**
       The tensor is fed through the pre-trained **EfficientNetB0** convolutional base (frozen during transfer learning). The base applies compound scaled inverted bottleneck convolutions (MBConv blocks) and global average pooling to yield a dense 1280-dimensional feature vector.
    5. **Classification Head:**
       The 1280-dimensional embedding passes through **Batch Normalization** to stabilize activations, followed by a **Dense layer of 512 units with ReLU activation**.
    6. **Softmax Output & Mapping:**
       The final Dense layer produces raw logits for all 49 classes, converted to normalized probabilities via **Softmax**. Output indices are accurately mapped to traffic sign names via the calibrated class dictionary.
    """)

with tab_arch:
    st.markdown("""
    ### 🏗️ EfficientNetB0 Transfer Learning Architecture
    
    ```text
    Input (128, 128, 3)
           │
           ▼
    EfficientNetB0 Backbone (ImageNet weights, include_top=False, pooling='avg')
    Output shape: (None, 1280) — 4.05M frozen parameters
           │
           ▼
    Batch Normalization Layer
    Output shape: (None, 1280)
           │
           ▼
    Dense Layer (512 units, ReLU activation)
    Output shape: (None, 512)
           │
           ▼
    Softmax Output Layer (49 units)
    Output shape: (None, 49)
    ```
    
    - **Total Parameters:** ~4.73 Million
    - **Trainable Parameters:** ~681,000 (Dense head & Batch Normalization)
    - **Backbone Parameters:** ~4.05 Million (Frozen EfficientNetB0 feature extractor)
    """)

with tab_dataset:
    st.markdown("""
    ### 📂 Dataset Information & Class Alignment
    
    - **Source Dataset:** Kaggle Traffic Sign Dataset Classification (`ahemateja19bec1025/traffic-sign-dataset-classification`).
    - **Total Original Samples:** 4,170 traffic sign images across 58 candidate categories.
    - **Pruning & Class Selection:** Classes containing fewer than 10 training instances were pruned (`SPLIT_MINIMUM_COUNT = 10`), retaining **49 robust classes**.
    - **Data Preprocessing During Training:**
      - Random rotation ($\pm 10^\circ$)
      - Width & height shift ($20\%$)
      - Shear ($10\%$) and Zoom ($20\%$)
      - Deterministic evaluation and test splits using seed $1$.
    - **Class-Index Calibration:** Keras `flow_from_dataframe` sorts string labels alphabetically (`0, 1, 10, 11, ..., 2, 20...`). The application properly maps these model output indices to their true human-readable labels.
    """)

with tab_tech:
    st.markdown("""
    ### 💻 Technology Stack
    
    | Component | Technology | Description |
    | :--- | :--- | :--- |
    | **Web Framework** | Streamlit | Responsive, high-performance UI with `@st.cache_resource` caching |
    | **Deep Learning** | TensorFlow / Keras | Model definition, weight management, and inference engine |
    | **Backbone Architecture** | EfficientNetB0 | Compound scaled CNN pre-trained on ImageNet |
    | **Image Processing** | Pillow (PIL) | Safe image ingestion, format conversion, and resizing |
    | **Numerical Processing**| NumPy | Fast multidimensional array and tensor manipulation |
    | **Serialization** | Joblib / Keras | Robust artifact persistence and weights loading |
    | **Deployment Platform** | Streamlit Cloud / Docker | Cloud-ready serverless execution with minimal dependencies |
    """)

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #64748B; font-size: 0.85rem; padding: 12px;'>"
    "University Deep Learning Final Project — Traffic Sign Classification Web App<br>"
    "Built with Streamlit & TensorFlow • Designed for Desktop & Mobile"
    "</div>",
    unsafe_allow_html=True
)

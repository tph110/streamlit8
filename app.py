#!/usr/bin/env python3
"""
DermScan AI - Skin Lesion Classification Application
8-class dermoscopic image classifier using EfficientNet-B4
Trained on combined ISIC2019 + HAM10000 + Fitzpatrick17k datasets

Performance Metrics:
- Macro F1: 0.8462
- Macro AUC: 0.9890
- Balanced Accuracy: 0.8488
"""

import streamlit as st
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import timm
import requests
from pathlib import Path
import io

# ============================================================
# Configuration
# ============================================================

# Class names and their full descriptions
CLASS_NAMES = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'scc', 'vasc']

CLASS_INFO = {
    'akiec': {
        'full_name': 'Actinic Keratosis / Bowen\'s Disease',
        'description': 'Pre-cancerous scaly patches caused by sun damage. Can progress to squamous cell carcinoma if untreated.',
        'risk_level': 'moderate',
        'recommendation': 'Schedule a dermatology appointment within 2-4 weeks for evaluation and possible treatment.'
    },
    'bcc': {
        'full_name': 'Basal Cell Carcinoma',
        'description': 'The most common type of skin cancer. Grows slowly and rarely spreads, but can cause local tissue damage.',
        'risk_level': 'high',
        'recommendation': 'Seek dermatology consultation within 1-2 weeks. Early treatment has excellent outcomes.'
    },
    'bkl': {
        'full_name': 'Benign Keratosis',
        'description': 'Non-cancerous skin growths including seborrheic keratoses and solar lentigines. Generally harmless.',
        'risk_level': 'low',
        'recommendation': 'Routine monitoring recommended. Consult a dermatologist if the lesion changes in appearance.'
    },
    'df': {
        'full_name': 'Dermatofibroma',
        'description': 'Benign fibrous skin nodule, often on the legs. Usually harmless and requires no treatment.',
        'risk_level': 'low',
        'recommendation': 'No urgent action needed. Monitor for any changes and discuss with your doctor at your next visit.'
    },
    'mel': {
        'full_name': 'Melanoma',
        'description': 'The most serious type of skin cancer. Can spread to other parts of the body if not caught early.',
        'risk_level': 'urgent',
        'recommendation': 'URGENT: Seek immediate dermatology evaluation. Early detection and treatment are critical for the best outcomes.'
    },
    'nv': {
        'full_name': 'Melanocytic Nevus (Mole)',
        'description': 'Common benign mole. Most people have 10-40 moles, which are generally harmless.',
        'risk_level': 'low',
        'recommendation': 'Normal finding. Monitor using the ABCDE rule and report any changes to your doctor.'
    },
    'scc': {
        'full_name': 'Squamous Cell Carcinoma',
        'description': 'Second most common skin cancer. Can grow quickly and has potential to spread if untreated.',
        'risk_level': 'high',
        'recommendation': 'Seek dermatology consultation within 1-2 weeks. Prompt treatment is important.'
    },
    'vasc': {
        'full_name': 'Vascular Lesion',
        'description': 'Blood vessel-related skin marks including angiomas and hemangiomas. Usually benign.',
        'risk_level': 'low',
        'recommendation': 'Generally benign. Consult a dermatologist if bleeding, growing, or causing concern.'
    }
}

RISK_COLORS = {
    'urgent': '#dc3545',   # Red
    'high': '#fd7e14',     # Orange
    'moderate': '#ffc107', # Yellow
    'low': '#28a745'       # Green
}

MODEL_URL = "https://huggingface.co/Skindoc/streamlit8/resolve/main/best_model_20251120_164117.pth"
MODEL_PATH = Path("best_model.pth")
IMG_SIZE = 384

# ============================================================
# Model Loading
# ============================================================

@st.cache_resource
def load_model():
    """Download and load the trained model."""
    # Download model if not present
    if not MODEL_PATH.exists():
        with st.spinner("Downloading model (this may take a moment on first run)..."):
            response = requests.get(MODEL_URL, stream=True)
            response.raise_for_status()
            with open(MODEL_PATH, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
    
    # Create model architecture
    model = timm.create_model('tf_efficientnet_b4', pretrained=False, num_classes=8)
    
    # Load trained weights
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model, device

# ============================================================
# Image Processing
# ============================================================

def get_transforms():
    """Get the validation transforms matching training configuration."""
    return transforms.Compose([
        transforms.Resize(int(IMG_SIZE * 1.05)),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

def get_tta_transforms():
    """Get Test-Time Augmentation transforms."""
    return [
        transforms.Compose([  # Original
            transforms.Resize(int(IMG_SIZE * 1.05)),
            transforms.CenterCrop(IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]),
        transforms.Compose([  # Horizontal flip
            transforms.Resize(int(IMG_SIZE * 1.05)),
            transforms.CenterCrop(IMG_SIZE),
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]),
        transforms.Compose([  # Vertical flip
            transforms.Resize(int(IMG_SIZE * 1.05)),
            transforms.CenterCrop(IMG_SIZE),
            transforms.RandomVerticalFlip(p=1.0),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]),
        transforms.Compose([  # 90 degree rotation
            transforms.Resize(int(IMG_SIZE * 1.05)),
            transforms.CenterCrop(IMG_SIZE),
            transforms.RandomRotation(degrees=(90, 90)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]),
    ]

def predict(model, device, image, use_tta=True):
    """Run prediction on an image with optional Test-Time Augmentation."""
    image = image.convert('RGB')
    
    with torch.no_grad():
        if use_tta:
            # Test-Time Augmentation: average predictions across augmented views
            tta_transforms = get_tta_transforms()
            all_probs = []
            
            for transform in tta_transforms:
                img_tensor = transform(image).unsqueeze(0).to(device)
                outputs = model(img_tensor)
                probs = F.softmax(outputs, dim=1).cpu().numpy()[0]
                all_probs.append(probs)
            
            # Average probabilities across all augmentations
            final_probs = np.mean(all_probs, axis=0)
        else:
            # Single prediction without TTA
            transform = get_transforms()
            img_tensor = transform(image).unsqueeze(0).to(device)
            outputs = model(img_tensor)
            final_probs = F.softmax(outputs, dim=1).cpu().numpy()[0]
    
    return final_probs

# ============================================================
# UI Components
# ============================================================

def display_results(probabilities):
    """Display classification results with clinical recommendations."""
    # Get predicted class
    pred_idx = np.argmax(probabilities)
    pred_class = CLASS_NAMES[pred_idx]
    pred_prob = probabilities[pred_idx]
    info = CLASS_INFO[pred_class]
    
    # Main prediction card
    st.markdown("---")
    st.subheader("Classification Result")
    
    # Risk-based styling
    risk_color = RISK_COLORS[info['risk_level']]
    risk_label = info['risk_level'].upper()
    
    # Display prediction
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown(f"### {info['full_name']}")
        st.markdown(f"**Classification:** `{pred_class.upper()}`")
        st.markdown(f"**Model Confidence:** {pred_prob*100:.1f}%")
        st.markdown(f"**Risk Level:** <span style='color:{risk_color}; font-weight:bold;'>{risk_label}</span>", unsafe_allow_html=True)
    
    with col2:
        # Confidence indicator
        if pred_prob >= 0.7:
            confidence_text = "High Confidence"
            confidence_color = "#28a745"
        elif pred_prob >= 0.5:
            confidence_text = "Moderate Confidence"
            confidence_color = "#ffc107"
        else:
            confidence_text = "Low Confidence"
            confidence_color = "#dc3545"
        
        st.markdown(f"<div style='text-align:center; padding:20px; background-color:{confidence_color}20; border-radius:10px; border:2px solid {confidence_color};'>"
                    f"<span style='color:{confidence_color}; font-size:1.2em; font-weight:bold;'>{confidence_text}</span></div>", 
                    unsafe_allow_html=True)
    
    # Description
    st.markdown("---")
    st.markdown("#### About This Condition")
    st.info(info['description'])
    
    # Clinical recommendation
    st.markdown("#### Recommended Action")
    if info['risk_level'] == 'urgent':
        st.error(f"🚨 {info['recommendation']}")
    elif info['risk_level'] == 'high':
        st.warning(f"⚠️ {info['recommendation']}")
    else:
        st.success(f"✓ {info['recommendation']}")
    
    # Probability chart
    st.markdown("---")
    st.markdown("#### All Class Probabilities")
    
    # Create data for chart
    chart_data = []
    for i, cls in enumerate(CLASS_NAMES):
        chart_data.append({
            'Class': f"{cls.upper()} ({CLASS_INFO[cls]['full_name'].split('/')[0].strip()[:20]})",
            'Probability': float(probabilities[i] * 100)
        })
    
    # Sort by probability descending
    chart_data.sort(key=lambda x: x['Probability'], reverse=True)
    
    # Display as horizontal bar chart using Streamlit
    import pandas as pd
    df = pd.DataFrame(chart_data)
    st.bar_chart(df.set_index('Class')['Probability'], horizontal=True)
    
    # Detailed probability table
    with st.expander("View Detailed Probabilities"):
        for item in chart_data:
            cls_code = item['Class'].split(' ')[0].lower()
            risk = CLASS_INFO[cls_code]['risk_level']
            color = RISK_COLORS[risk]
            st.markdown(f"<span style='color:{color};'>●</span> **{item['Class']}**: {item['Probability']:.2f}%", 
                        unsafe_allow_html=True)

def show_disclaimer():
    """Display medical disclaimer."""
    st.markdown("""
    <div style='background-color:#fff3cd; padding:15px; border-radius:5px; border-left:5px solid #ffc107; margin-bottom:20px;'>
    <strong>⚠️ Medical Disclaimer</strong><br>
    This AI tool is designed to assist healthcare professionals and is <strong>NOT</strong> a substitute for professional medical advice, diagnosis, or treatment.
    <ul>
    <li>Always consult a qualified dermatologist for any skin concerns</li>
    <li>Do not make medical decisions based solely on this tool's output</li>
    <li>This tool is intended for educational and research purposes</li>
    <li>In case of urgent concerns, seek immediate medical attention</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

def show_model_info():
    """Display model performance information."""
    with st.expander("ℹ️ Model Information & Performance"):
        st.markdown("""
        **Model Architecture:** EfficientNet-B4  
        **Training Data:** Combined ISIC2019 + HAM10000 + Fitzpatrick17k datasets  
        **Image Input Size:** 384×384 pixels  
        
        **Validation Performance:**
        | Metric | Score |
        |--------|-------|
        | Macro F1 | 0.8462 |
        | Macro AUC | 0.9890 |
        | Balanced Accuracy | 0.8488 |
        
        **Per-Class Performance (F1 Score):**
        | Class | Full Name | F1 Score |
        |-------|-----------|----------|
        | NV | Melanocytic Nevus | 0.9501 |
        | BCC | Basal Cell Carcinoma | 0.9105 |
        | BKL | Benign Keratosis | 0.9008 |
        | DF | Dermatofibroma | 0.8800 |
        | VASC | Vascular Lesion | 0.8761 |
        | MEL | Melanoma | 0.8580 |
        | AKIEC | Actinic Keratosis | 0.7201 |
        | SCC | Squamous Cell Carcinoma | 0.6742 |
        
        **Safety Features:**
        - Clinical safety weighting applied during training (2× for Melanoma, 1.5× for SCC)
        - Test-Time Augmentation for improved prediction stability
        - Focal loss to handle class imbalance
        """)

# ============================================================
# Main Application
# ============================================================

def main():
    # Page configuration
    st.set_page_config(
        page_title="DermScan AI - Skin Lesion Classifier",
        page_icon="🔬",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Header
    st.title("🔬 DermScan AI")
    st.markdown("### Dermoscopic Skin Lesion Classification")
    st.markdown("*AI-powered analysis of skin lesions using deep learning*")
    
    # Show disclaimer
    show_disclaimer()
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        use_tta = st.checkbox(
            "Enable Test-Time Augmentation (TTA)",
            value=True,
            help="Averages predictions across multiple augmented views for more robust results. Slightly slower but more accurate."
        )
        
        st.markdown("---")
        show_model_info()
        
        st.markdown("---")
        st.markdown("""
        **Classes Detected:**
        - 🟢 NV - Melanocytic Nevus
        - 🟢 BKL - Benign Keratosis
        - 🟢 DF - Dermatofibroma
        - 🟢 VASC - Vascular Lesion
        - 🟡 AKIEC - Actinic Keratosis
        - 🟠 BCC - Basal Cell Carcinoma
        - 🟠 SCC - Squamous Cell Carcinoma
        - 🔴 MEL - Melanoma
        """)
        
        st.markdown("---")
        st.markdown("**Developer:** Dr. Tom Hutchinson")
        st.markdown("**Version:** 1.0.0")
    
    # Load model
    try:
        model, device = load_model()
        device_name = "GPU (CUDA)" if device.type == 'cuda' else "CPU"
        st.sidebar.success(f"✓ Model loaded on {device_name}")
    except Exception as e:
        st.error(f"Failed to load model: {str(e)}")
        st.stop()
    
    # Main content area
    st.markdown("---")
    st.subheader("📤 Upload Dermoscopic Image")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Choose a dermoscopic image...",
            type=['jpg', 'jpeg', 'png', 'bmp'],
            help="Upload a dermoscopic (dermatoscope) image of a skin lesion"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_container_width=True)
            
            # Image info
            st.markdown(f"**Image Size:** {image.size[0]} × {image.size[1]} pixels")
            st.markdown(f"**Format:** {image.format or 'Unknown'}")
    
    with col2:
        if uploaded_file is not None:
            # Analyze button
            if st.button("🔍 Analyze Image", type="primary", use_container_width=True):
                with st.spinner("Analyzing image..."):
                    # Run prediction
                    probabilities = predict(model, device, image, use_tta=use_tta)
                
                # Display results
                display_results(probabilities)
        else:
            st.info("👈 Upload a dermoscopic image to begin analysis")
            
            # Example usage instructions
            st.markdown("""
            **How to use:**
            1. Upload a dermoscopic image using the file uploader
            2. Click "Analyze Image" to run the AI classification
            3. Review the results and clinical recommendations
            
            **Best practices:**
            - Use high-quality dermoscopic images
            - Ensure the lesion is centered and in focus
            - Images should ideally be taken with a dermatoscope
            """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align:center; color:#666; font-size:0.9em;'>
    <p>This tool uses deep learning to classify dermoscopic images into 8 categories of skin lesions.</p>
    <p>For research and educational purposes only. Always consult a qualified healthcare professional.</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()

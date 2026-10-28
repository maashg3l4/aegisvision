import streamlit as st
import cv2
import numpy as np
from PIL import Image
import json
import urllib.request
import os
import sys
import time
from ultralytics import YOLO

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.report_assistant.generator import generate_report, load_knowledge_base

# --- CUSTOM CSS FOR MIT-LEVEL UI ---
st.markdown("""
<style>
    /* Fade-in animation */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .stApp { background-color: #0e1117; }
    
    /* Title Styling */
    h1 { font-family: 'Inter', sans-serif; font-weight: 700; letter-spacing: -1px; }
    
    /* Report Card */
    .report-card {
        background: rgba(30, 41, 59, 0.7);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 24px;
        margin-top: 20px;
        animation: fadeIn 0.6s ease-out;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    }
    .report-title { color: #60a5fa; font-size: 14px; font-weight: 600; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 12px; }
    .report-text { color: #e2e8f0; font-size: 15px; line-height: 1.6; }
    .action-item { color: #f87171; font-weight: 600; margin-top: 10px; display: block; }
    
    /* Hide Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# --- LOAD MODELS ---
@st.cache_resource
def load_models():
    model_url = "https://github.com/maashg3l4/aegisvision/releases/download/v1.0.1/best.pt"
    policy_url = "https://github.com/maashg3l4/aegisvision/releases/download/v1.0.0/routing_policy_v2.json"
    
    if not os.path.exists("/tmp/best.pt"):
        urllib.request.urlretrieve(model_url, "/tmp/best.pt")
    if not os.path.exists("/tmp/policy.json"):
        urllib.request.urlretrieve(policy_url, "/tmp/policy.json")
        
    model = YOLO("/tmp/best.pt")
    with open("/tmp/policy.json") as f:
        policy = json.load(f)
    return model, policy

model, policy = load_models()
kb = load_knowledge_base()

# --- FEATURE EXTRACTION ---
FEATURES = ['max_conf','mean_conf','std_conf','p90_conf','top1_gap','conf_entropy',
            'n_preds','n_mid','n_high','mean_area','min_area','mean_ar','std_ar',
            'small_ratio','unique_classes_pred']

def extract_features(results):
    boxes = results.boxes
    if len(boxes) == 0:
        return {f: 0.0 for f in FEATURES}
    scores = boxes.conf.cpu().numpy()
    classes = boxes.cls.cpu().numpy()
    xywh = boxes.xywh.cpu().numpy()
    areas = xywh[:, 2] * xywh[:, 3]
    ars = xywh[:, 2] / np.maximum(xywh[:, 3], 1e-6)
    hist, _ = np.histogram(scores, bins=10, range=(0, 1))
    return {
        'max_conf': float(scores.max()), 'mean_conf': float(scores.mean()),
        'std_conf': float(scores.std()), 'p90_conf': float(np.percentile(scores, 90)),
        'top1_gap': float(scores[0] - scores.mean()) if len(scores) > 1 else 0.0,
        'conf_entropy': float(-np.sum((hist / np.sum(hist)) * np.log2((hist / np.sum(hist)) + 1e-9))),
        'n_preds': len(scores), 'n_mid': int(((scores > 0.2) & (scores <= 0.5)).sum()),
        'n_high': int((scores > 0.5).sum()), 'mean_area': float(areas.mean()),
        'min_area': float(areas.min()), 'mean_ar': float(ars.mean()),
        'std_ar': float(ars.std()), 'small_ratio': float((areas < 500).sum()) / len(areas),
        'unique_classes_pred': int(len(set(classes)))
    }

# --- UI ---
st.title("AegisVision: Defect Inspection Assistant")
st.markdown("<p style='color: #94a3b8; font-size: 16px;'>Upload an image for mathematical risk assessment.</p>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    
    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Original Upload", use_column_width=True)
    
    if st.button("Run Inspection", use_container_width=True):
        with st.spinner("Analyzing with YOLO11s and 16-Feature Policy..."):
            img_np = np.array(image)
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            results = model(img_bgr, conf=0.001, verbose=False)[0]
            
            # --- OOD SHIELD ---
            is_ood = False
            energy_score = 0.0
            if len(results.boxes) > 0:
                probs = results.boxes.conf.cpu().numpy()
                probs = np.clip(probs, 1e-7, 1 - 1e-7)
                logits = np.log(probs / (1 - probs))
                energies = -np.log(np.sum(np.exp(logits)))
                energy_score = float(np.max(energies))
                if energy_score > 0.0:
                    is_ood = True
            
            # --- VISUAL FILTER (Only show boxes > 0.40 conf to user) ---
            vis_results = results
            if len(results.boxes) > 0:
                high_conf_mask = results.boxes.conf > 0.40
                vis_results = results[high_conf_mask]
            
            # Draw boxes
            if is_ood:
                annotated_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            else:
                annotated_bgr = vis_results.plot(conf=True, line_width=1)
                annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
                
            with col2:
                st.image(annotated_rgb, caption="Filtered Detections (Conf > 0.40)", use_column_width=True)
                
            # --- ROUTING LOGIC ---
            feats_dict = extract_features(results)
            x = np.array([feats_dict[f] for f in FEATURES])
            mu = np.array(policy["scaler"]["mean"])[:15]
            sd = np.array(policy["scaler"]["scale"])[:15]
            w = np.array(policy["lr_model"]["coefficients"])[:15]
            b = policy["lr_model"]["intercept"]
            
            z = float(((x - mu) / sd) @ w + b)
            danger_score = 1.0 / (1.0 + np.exp(-z))
            
            # --- DECISION UI ---
            st.markdown("---")
            # --- SAFETY OVERRIDE LOGIC ---
            CRITICAL_CLASSES = ["crazing", "rolled-in_scale", "scratches"]
            top_class_id = int(results.boxes.cls[0].item()) if len(results.boxes) > 0 else -1
            CLASS_NAMES = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]
            top_class_name = CLASS_NAMES[top_class_id] if top_class_id >= 0 and top_class_id < len(CLASS_NAMES) else "unknown"
            
            force_review = top_class_name in CRITICAL_CLASSES

            if is_ood:
                st.error(f"OUT-OF-DISTRIBUTION DETECTED (Energy: {energy_score:.2f}) — MANUAL REVIEW REQUIRED")
            elif force_review:
                st.error(f"HUMAN REVIEW REQUIRED (Critical Class: {top_class_name.upper()} detected) — Safety Override")
            else:
                if danger_score > 0.80:
                    st.error(f"HUMAN REVIEW REQUIRED (Risk Score: {danger_score:.4f})")
                else:
                    st.success(f"AUTO-ACCEPTED (Risk Score: {danger_score:.4f})")
            
            # --- REPORT ASSISTANT (Custom HTML Card) ---
            if not is_ood and len(results.boxes) > 0:
                top_class_id = int(results.boxes.cls[0].item())
                CLASS_NAMES = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]
                top_class_name = CLASS_NAMES[top_class_id] if top_class_id < len(CLASS_NAMES) else "unknown"

                try:
                    report_text = generate_report(
                        defect_class=top_class_name,
                        confidence=float(results.boxes.conf[0].item()),
                        danger_score=danger_score,
                        kb=kb,
                        force_review=force_review
                    )

                    # Format report into HTML
                    lines = report_text.strip().split("
")
                    html_content = "<div class='report-card'>"
                    html_content += "<div class='report-title'>Aegis Report Assistant</div>"
                    html_content += "<div class='report-text'>"

                    for line in lines:
                        if "RECOMMENDED ACTION" in line or "Inspect" in line:
                            html_content += f"<span class='action-item'>{line.strip()}</span>"
                        elif "===" in line:
                            continue
                        else:
                            html_content += f"{line.strip()}<br>"

                    html_content += "</div></div>"
                    st.markdown(html_content, unsafe_allow_html=True)
                    
                except Exception as e:
                    st.warning(f"Report Assistant error: {e}")

import streamlit as st
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.report_assistant.generator import generate_report, load_knowledge_base
import cv2
import numpy as np
from PIL import Image
import json
import urllib.request
import os
from ultralytics import YOLO

# --- Auto-Download Models from GitHub ---
def download_file(url, local_path):
    if not os.path.exists(local_path):
        with st.spinner(f"Downloading {os.path.basename(local_path)}..."):
            urllib.request.urlretrieve(url, local_path)

MODEL_PATH = "/tmp/best.pt"
POLICY_PATH = "/tmp/routing_policy_v2.json"
MODEL_URL = "https://github.com/maashg3l4/aegisvision/releases/download/v1.0.1/best.pt"
POLICY_URL = "https://github.com/maashg3l4/aegisvision/releases/download/v1.0.0/routing_policy_v2.json"

@st.cache_resource
def load_models():
    download_file(MODEL_URL, MODEL_PATH)
    download_file(POLICY_URL, POLICY_PATH)
    
    model = YOLO(MODEL_PATH)
    with open(POLICY_PATH) as f:
        policy = json.load(f)
    return model, policy

model, policy = load_models()

# --- 15 Live Features ---
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

# --- Streamlit UI ---
st.set_page_config(page_title="AegisVision", layout="wide")
st.title("🛡️ AegisVision: Defect Inspection Assistant")
st.markdown("Upload an image to get an instant risk assessment.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    if st.button("Analyze Image"):
        with st.spinner("Running YOLO11s and Routing Policy..."):
            img_np = np.array(image)
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            results = model(img_bgr, conf=0.001, verbose=False)[0]
            
            # --- OOD SHIELD: Energy-Based Detection ---
            is_ood = False
            energy_score = 0.0
            if len(results.boxes) > 0:
                probs = results.boxes.conf.cpu().numpy()
                probs = np.clip(probs, 1e-7, 1 - 1e-7)
                logits = np.log(probs / (1 - probs))
                energies = -np.log(np.sum(np.exp(logits)))
                energy_score = float(np.max(energies))
                
                # If energy is positive, the model is hallucinating (OOD)
                if energy_score > 0.0:
                    is_ood = True
            
            # --- VISUALIZATION: Draw bounding boxes ---
            # If OOD, we do not draw boxes. If normal, we draw them.
            if is_ood:
                st.warning("⚠️ System Confused: Out-of-Distribution Image Detected. Boxes suppressed.")
                annotated_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB) # Show original
            else:
                annotated_bgr = results.plot(conf=True, line_width=1)
            annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
            
            st.subheader("YOLO11s Detections")
            st.image(annotated_rgb, caption="Detected Defects", use_column_width=True)
            # -------------------------------------------
            
            feats_dict = extract_features(results)
            x = np.array([feats_dict[f] for f in FEATURES])
            
            mu = np.array(policy["scaler"]["mean"])[:15]
            sd = np.array(policy["scaler"]["scale"])[:15]
            w = np.array(policy["lr_model"]["coefficients"])[:15]
            b = policy["lr_model"]["intercept"]
            
            z = float(((x - mu) / sd) @ w + b)
            danger_score = 1.0 / (1.0 + np.exp(-z))
            
            # --- OOD OVERRIDE ---
            if is_ood:
                decision = "HUMAN_REVIEW"
                st.error(f"🔴 **HUMAN REVIEW REQUIRED** (OOD Shield Triggered - Energy: {energy_score:.2f})")
            else:
                decision = "HUMAN_REVIEW" if danger_score > 0.80 else "AUTO_ACCEPT"
                if decision == "HUMAN_REVIEW":
                    st.error(f"🔴 **HUMAN REVIEW REQUIRED** (Risk Score: {danger_score:.4f})")
                else:
                    st.success(f"🟢 **AUTO-ACCEPTED** (Risk Score: {danger_score:.4f})")
                
            st.json({
                "decision": decision,
                "danger_score": round(danger_score, 4),
                "num_predictions": feats_dict["n_preds"]
            })
            
            # --- REPORT ASSISTANT INTEGRATION ---
            if not is_ood and len(results.boxes) > 0:
                # Get the most confident defect class name
                top_class_id = int(results.boxes.cls[0].item())
                # NEU-DET classes mapping (ensure this matches your model)
                CLASS_NAMES = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]
                top_class_name = CLASS_NAMES[top_class_id] if top_class_id < len(CLASS_NAMES) else "unknown"
                
                try:
                    kb = load_knowledge_base()
                    report = generate_report(
                        defect_class=top_class_name,
                        confidence=float(results.boxes.conf[0].item()),
                        danger_score=danger_score,
                        kb=kb
                    )
                    st.subheader("🤖 Aegis Report Assistant")
                    st.info(report)
                except Exception as e:
                    st.warning(f"Report Assistant error: {e}")

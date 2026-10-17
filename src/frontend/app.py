import streamlit as st
import cv2
import numpy as np
from PIL import Image
import json
from ultralytics import YOLO
from scipy.stats import entropy as scipy_entropy

# --- Load Models (Cached for performance) ---
@st.cache_resource
def load_models():
    # Update these paths if you move them to a cloud storage bucket
    yolo_path = "/content/drive/MyDrive/AegisVision/runs/baseline_yolo11s/weights/best.pt"
    policy_path = "/content/drive/MyDrive/AegisVision/policies/routing_policy_v2.json"
    
    model = YOLO(yolo_path)
    with open(policy_path) as f:
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
        'conf_entropy': float(scipy_entropy(hist + 1e-9)),
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
            # Run YOLO
            img_np = np.array(image)
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            results = model(img_bgr, conf=0.001, verbose=False)[0]
            
            # Extract Features & Score
            feats_dict = extract_features(results)
            x = np.array([feats_dict[f] for f in FEATURES])
            
            # 15 features slicing
            mu = np.array(policy["scaler"]["mean"])[:15]
            sd = np.array(policy["scaler"]["scale"])[:15]
            w = np.array(policy["lr_model"]["coefficients"])[:15]
            b = policy["lr_model"]["intercept"]
            
            z = float(((x - mu) / sd) @ w + b)
            danger_score = 1.0 / (1.0 + np.exp(-z))
            
            decision = "HUMAN_REVIEW" if danger_score > 0.80 else "AUTO_ACCEPT"
            
            # Output
            if decision == "HUMAN_REVIEW":
                st.error(f"🔴 **HUMAN REVIEW REQUIRED** (Risk Score: {danger_score:.4f})")
            else:
                st.success(f"🟢 **AUTO-ACCEPTED** (Risk Score: {danger_score:.4f})")
                
            st.json({
                "decision": decision,
                "danger_score": round(danger_score, 4),
                "num_predictions": feats_dict["n_preds"]
            })

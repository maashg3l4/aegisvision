"""AegisVision FastAPI Production Server."""
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import numpy as np
import json
import cv2
import io
from PIL import Image
from ultralytics import YOLO
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from api.live_features import extract_live_features, FEATURES

app = FastAPI(title="AegisVision API", version="1.0.0")

# Load models at startup
YOLO_PATH = "/content/drive/MyDrive/AegisVision/runs/baseline_yolo11s/weights/best.pt"
POLICY_PATH = "/content/drive/MyDrive/AegisVision/policies/routing_policy_v2.json"

model = YOLO(YOLO_PATH)
with open(POLICY_PATH) as f:
    policy = json.load(f)

# Update scaler and weights to exclude the 16th feature (class_mismatch)
# We must slice the arrays to match our 15 live features
scaler_mean = np.array(policy["scaler"]["mean"])[:15]
scaler_scale = np.array(policy["scaler"]["scale"])[:15]
lr_coef = np.array(policy["lr_model"]["coefficients"])[:15]
lr_intercept = policy["lr_model"]["intercept"]

@app.get("/health")
def health():
    return {"status": "healthy", "model": "YOLO11s", "policy": "v2"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # 1. Read image
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # 2. Run YOLO
    results = model(img, conf=0.001, verbose=False)[0]
    
    # 3. Extract features
    feats_dict = extract_live_features(results)
    x = np.array([feats_dict[f] for f in FEATURES])
    
    # 4. Compute Danger Score
    z = float(((x - scaler_mean) / scaler_scale) @ lr_coef + lr_intercept)
    danger_score = 1.0 / (1.0 + np.exp(-z))
    
    # 5. Routing Decision (Budget = 20% -> Threshold = 0.80 danger score)
    # If danger > 0.80, it goes to the top 20% riskiest images
    decision = "HUMAN_REVIEW" if danger_score > 0.80 else "AUTO_ACCEPT"
    
    # 6. Format boxes
    boxes_list = []
    for box in results.boxes:
        boxes_list.append({
            "class": int(box.cls[0]),
            "confidence": float(box.conf[0]),
            "bbox": [float(v) for v in box.xyxy[0]]
        })
        
    return JSONResponse(content={
        "decision": decision,
        "danger_score": round(danger_score, 4),
        "num_predictions": feats_dict["n_preds"],
        "boxes": boxes_list
    })

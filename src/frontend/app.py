import streamlit as st
import cv2
import numpy as np
from PIL import Image
import json
import urllib.request
import os
import sys
from ultralytics import YOLO
from jsonschema import validate, ValidationError
from groq import Groq

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.report_assistant.generator import generate_report, load_knowledge_base
from src.flywheel.manager import FlywheelManager

# --- GLOBAL CONSTANTS ---
CLASS_NAMES = [
    'crazing', 'inclusion', 'patches', 'pitted_surface', 'rolled-in_scale', 'scratches',
    'punching_hole', 'welding_line', 'crescent_gap', 'water_spot', 'oil_spot',
    'silk_spot', 'rolled_pit', 'crease', 'waist_folding', 'dents', 'macular'
]
CRITICAL_CLASSES = ['crazing', 'rolled-in_scale', 'scratches', 'crease']
DECISION_ACCEPT = "ACCEPT"
DECISION_HUMAN_REVIEW = "HUMAN_REVIEW"

# --- CUSTOM CSS ---
st.markdown("""
<style>
    @keyframes fadeIn { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }
    .stApp { background-color: #0e1117; }
    h1 { font-family: 'Inter', sans-serif; font-weight: 700; letter-spacing: -1px; }
    .report-card { background: rgba(30, 41, 59, 0.7); backdrop-filter: blur(10px); border: 1px solid rgba(255, 255, 255, 0.1); border-radius: 12px; padding: 24px; margin-top: 20px; animation: fadeIn 0.6s ease-out; box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3); }
    .report-title { color: #60a5fa; font-size: 14px; font-weight: 600; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 12px; }
    .report-text { color: #e2e8f0; font-size: 15px; line-height: 1.6; }
    .action-item { color: #f87171; font-weight: 600; margin-top: 10px; display: block; }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# --- LOAD MODELS ---
@st.cache_resource
def load_models():
    model_url = "https://github.com/maashg3l4/aegisvision/releases/download/v4.2.1/best.pt"
    policy_url = "https://raw.githubusercontent.com/maashg3l4/aegisvision/main/src/policy/routing_policy_v3.json"
    if not os.path.exists("/tmp/best.pt"): urllib.request.urlretrieve(model_url, "/tmp/best.pt")
    if not os.path.exists("/tmp/policy.json"): urllib.request.urlretrieve(policy_url, "/tmp/policy.json")
    model = YOLO("/tmp/best.pt")
    with open("/tmp/policy.json") as f: policy = json.load(f)
    return model, policy

model, policy = load_models()
kb = load_knowledge_base()
try:
    fm = FlywheelManager()
    flywheel_available = True
except Exception:
    flywheel_available = False

# --- FEATURE EXTRACTION ---
FEATURES = ['max_conf','mean_conf','std_conf','p90_conf','top1_gap','conf_entropy', 'n_preds','n_mid','n_high','mean_area','min_area','mean_ar','std_ar', 'small_ratio','unique_classes_pred']

def extract_features(results):
    boxes = results.boxes
    if len(boxes) == 0: return {f: 0.0 for f in FEATURES}
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

# --- SINGLE SOURCE OF TRUTH ---
def route_inspection(raw_results, policy, is_ood, energy_score):
    reason_codes = []
    if is_ood:
        reason_codes.append("OUT_OF_DOMAIN")
        return {"decision": DECISION_HUMAN_REVIEW, "reason_codes": reason_codes, "danger_score": 0.5, "top_class_name": "unknown", "force_review": True}
    
    if len(raw_results.boxes) == 0:
        reason_codes.append("NO_DETECTIONS")
        return {"decision": DECISION_HUMAN_REVIEW, "reason_codes": reason_codes, "danger_score": 0.0, "top_class_name": "none", "force_review": False}
        
    feats = extract_features(raw_results)
    x = np.array([feats[f] for f in FEATURES])
    mu, sd = np.array(policy['scaler']['mean'])[:15], np.array(policy['scaler']['scale'])[:15]
    w, b = np.array(policy['lr_model']['coefficients'])[:15], policy['lr_model']['intercept']
    danger_score = 1.0 / (1.0 + np.exp(-float(((x - mu) / sd) @ w + b)))
    
    top_cls = CLASS_NAMES[int(raw_results.boxes.cls[0].item())] if int(raw_results.boxes.cls[0].item()) < len(CLASS_NAMES) else 'unknown'
    max_conf = float(raw_results.boxes.conf[0].item())
    
    if top_cls in CRITICAL_CLASSES: reason_codes.append("CRITICAL_CLASS")
    if max_conf < 0.60: reason_codes.append("LOW_CONFIDENCE")
    if danger_score > 0.80: reason_codes.append("HIGH_RISK")
    
    decision = DECISION_HUMAN_REVIEW if reason_codes else DECISION_ACCEPT
    return {"decision": decision, "reason_codes": reason_codes, "danger_score": danger_score, "top_class_name": top_cls, "force_review": top_cls in CRITICAL_CLASSES}

# --- STATE MANAGEMENT ---
if 'messages' not in st.session_state: st.session_state.messages = []
if 'inspection_results' not in st.session_state: st.session_state.inspection_results = None

# --- UI ---
st.title('AegisVision: Defect Inspection Assistant')
groq_key = st.secrets.get('GROQ_API_KEY', '')
if not groq_key:
    with st.sidebar:
        st.header('Configuration')
        st.text_input('Groq API Key:', type='password', key='groq_api_key')
    groq_key = st.session_state.get('groq_api_key', '')

uploaded_file = st.file_uploader('', type=['jpg', 'jpeg', 'png'])

if st.button('Run Inspection', use_container_width=True) and uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    img_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    with st.spinner('Analyzing...'):
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        is_color_ood = np.mean(hsv[:, :, 1]) > 30
        
        raw_results = model(img_bgr, conf=0.001, agnostic_nms=True, verbose=False)[0]
        
        is_energy_ood, energy_score = False, 0.0
        if len(raw_results.boxes) > 0:
            probs = np.clip(raw_results.boxes.conf.cpu().numpy(), 1e-7, 1 - 1e-7)
            energies = -np.log(np.sum(np.exp(np.log(probs / (1 - probs)))))
            energy_score = float(np.max(energies))
            if energy_score > 0.0: is_energy_ood = True
        
        is_ood = is_color_ood or is_energy_ood
        route = route_inspection(raw_results, policy, is_ood, energy_score)
        
        # Visual Evidence: Draw boxes if NOT OOD (even low conf)
        if is_ood:
            annotated_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        else:
            annotated_bgr = img_bgr.copy()
            for box in raw_results.boxes:
                conf = float(box.conf[0])
                if conf < 0.05: continue # Ignore pure noise, but show low conf
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                color = (0, 165, 255) if conf < 0.60 else (0, 255, 0) # Orange if low conf, Green if high
                label = f"{CLASS_NAMES[int(box.cls[0])]} {conf:.2f}"
                cv2.rectangle(annotated_bgr, (x1, y1), (x2, y2), color, 2)
                cv2.putText(annotated_bgr, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

        st.session_state.inspection_results = {
            'image': image, 'results': raw_results, 'is_ood': bool(is_ood), 'energy_score': float(energy_score),
            'decision': str(route['decision']), 'reason_codes': route['reason_codes'], 'danger_score': float(route['danger_score']),
            'top_class_name': str(route['top_class_name']), 'force_review': bool(route['force_review']), 'annotated_rgb': annotated_rgb
        }
        st.session_state.messages = []

# --- RENDER RESULTS ---
if st.session_state.inspection_results is not None:
    res = st.session_state.inspection_results
    col1, col2 = st.columns(2)
    with col1: st.image(res['image'], caption='Original Upload', use_column_width=True)
    with col2: 
        caption = "No defect localized. Image flagged as unfamiliar." if res['is_ood'] else "Detections (Orange = Low Confidence)"
        st.image(res['annotated_rgb'], caption=caption, use_column_width=True)

    st.markdown('---')
    reasons_str = ", ".join(res['reason_codes'])
    if res['is_ood']:
        st.error(f"⚠️ UNFAMILIAR IMAGE: Out-of-Distribution detected. Manual inspection required. (Reasons: {reasons_str})")
    elif res['decision'] == DECISION_HUMAN_REVIEW:
        st.warning(f"HUMAN REVIEW REQUIRED: {reasons_str}")
    else:
        st.success(f"AUTO-ACCEPTED: Clean part detected.")

    # Report Assistant (Only if not OOD and a class is known)
    if not res['is_ood'] and res['top_class_name'] != 'unknown' and len(res['results'].boxes) > 0:
        try:
            report_text = generate_report(res['top_class_name'], float(res['results'].boxes.conf[0].item()), res['danger_score'], kb, res['force_review'])
            lines = report_text.strip().split('\n')
            html = "<div class='report-card'><div class='report-title'>Aegis Report Assistant</div><div class='report-text'>"
            for line in lines:
                if 'RECOMMENDED ACTION' in line or 'Inspect' in line: html += f"<span class='action-item'>{line.strip()}</span>"
                elif '===' in line: continue
                else: html += f"{line.strip()}<br>"
            html += '</div></div>'
            st.markdown(html, unsafe_allow_html=True)
        except Exception as e: st.warning(f'Report Assistant error: {e}')

    # Feedback
    st.markdown('---')
    st.subheader('Inspector Feedback')
    temp_img_path = '/tmp/temp_upload.jpg'
    cv2.imwrite(temp_img_path, cv2.cvtColor(np.array(res['image']), cv2.COLOR_RGB2BGR))
    
    col1, col2, col3 = st.columns(3)
    conf_val = float(res['results'].boxes.conf[0].item()) if len(res['results'].boxes) > 0 else 0.0
    with col1:
        if st.button('✅ Correct', use_container_width=True) and flywheel_available:
            try: fm.save_feedback(temp_img_path, res['top_class_name'], conf_val, res['danger_score'], 'CORRECT'); st.success('Saved!')
            except Exception as e: st.error(f"Error: {e}")
    with col2:
        if st.button('❌ False Positive', use_container_width=True) and flywheel_available:
            try: fm.save_feedback(temp_img_path, res['top_class_name'], conf_val, res['danger_score'], 'FALSE_POSITIVE'); st.success('Saved!')
            except Exception as e: st.error(f"Error: {e}")
    with col3:
        if st.button('⚠️ Missed Defect', use_container_width=True) and flywheel_available:
            try: fm.save_feedback(temp_img_path, res['top_class_name'], conf_val, res['danger_score'], 'MISSED_DEFECT'); st.success('Saved!')
            except Exception as e: st.error(f"Error: {e}")

# --- CHAT ---
if st.session_state.inspection_results is not None:
    st.markdown('---')
    st.subheader('Ask Aegis')
    for message in st.session_state.messages:
        with st.chat_message(message['role']): st.markdown(message['content'])

    if not groq_key: st.info('Enter Groq API Key in sidebar.')
    else:
        st.markdown('**Suggested Questions:**')
        col1, col2, col3 = st.columns(3)
        suggested_q = None
        with col1:
            if st.button('What defect was found?', use_container_width=True): suggested_q = 'What defect was found?'
        with col2:
            if st.button('Why is this flagged?', use_container_width=True): suggested_q = 'Why is this image sent for human review?'
        with col3:
            if st.button('What is the risk score?', use_container_width=True): suggested_q = 'What is the risk score of this inspection?'
                
        user_question = suggested_q 
        chat_input_val = st.chat_input('Ask about this inspection...')
        if chat_input_val: user_question = chat_input_val

        if user_question:
            st.session_state.messages.append({'role': 'user', 'content': user_question})
            with st.chat_message('user'): st.markdown(user_question)

            with st.chat_message('assistant'):
                with st.spinner('Thinking...'):
                    try:
                        client = Groq(api_key=groq_key)
                        
                        # DETERMINISTIC PREFIX
                        review_status = "yes" if st.session_state.inspection_results['decision'] == DECISION_HUMAN_REVIEW else "no"
                        prefix = f"Decision: {st.session_state.inspection_results['decision']}. Reasons: {', '.join(st.session_state.inspection_results['reason_codes'])}. Human review: {review_status}. Risk Score: {round(st.session_state.inspection_results['danger_score'], 2)}.\n\n"
                        
                        inspection_record = {
                            'inspection_id': 'live',
                            'detections': [{'class_name': CLASS_NAMES[int(b.cls[0])], 'confidence': round(float(b.conf[0]), 2)} for b in st.session_state.inspection_results['results'].boxes][:5],
                            'uncertainty_assessment': {'risk_score': round(st.session_state.inspection_results['danger_score'], 2), 'is_ood': st.session_state.inspection_results['is_ood']},
                            'routing_decision': st.session_state.inspection_results['decision'],
                            'reason_codes': st.session_state.inspection_results['reason_codes']
                        }
                        
                        system_prompt = """You are AegisVision. Answer the user's question using ONLY the provided JSON.
RULES:
1. The deterministic prefix provided in the prompt is the absolute truth. Do not contradict it.
2. If the decision is HUMAN_REVIEW because of OUT_OF_DOMAIN, explain that the image does not look like the training data.
3. If the decision is HUMAN_REVIEW because of LOW_CONFIDENCE or CRITICAL_CLASS, explain that the model found a defect but requires human validation.
4. Do not invent thresholds. If a fact is not in the JSON, say "I do not have that information."
5. Output valid JSON with 'answer_text', 'is_abstained', 'abstain_reason'.
"""
                        user_prompt = f"JSON: {json.dumps(inspection_record)}\n\nDeterministic Truth: {prefix}\n\nQuestion: {user_question}\n\nAnswer in JSON format."
                        
                        chat_completion = client.chat.completions.create(
                            messages=[{'role': 'system', 'content': system_prompt}, {'role': 'user', 'content': user_prompt}],
                            model='openai/gpt-oss-20b', temperature=0.0, response_format={'type': 'json_object'}
                        )
                        res_json = json.loads(chat_completion.choices[0].message.content)
                        
                        if res_json.get('is_abstained', False):
                            final_answer = f"⚠️ **Aegis Abstained**: {res_json.get('abstain_reason', 'Insufficient evidence.')}"
                        else:
                            final_answer = prefix + res_json.get('answer_text', 'No answer generated.')
                        
                        st.markdown(final_answer)
                        st.session_state.messages.append({'role': 'assistant', 'content': final_answer})
                    except Exception as e:
                        st.error(f'Chat Error: {e}')

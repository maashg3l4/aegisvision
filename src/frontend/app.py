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
from src.report_assistant.vlm_prompt import SYSTEM_PROMPT, build_prompt
from src.flywheel.manager import FlywheelManager

# --- GLOBAL CONSTANTS ---
CLASS_NAMES = [
    'crazing', 'inclusion', 'patches', 'pitted_surface', 'rolled-in_scale', 'scratches',
    'punching_hole', 'welding_line', 'crescent_gap', 'water_spot', 'oil_spot',
    'silk_spot', 'rolled_pit', 'crease', 'waist_folding'
]
CRITICAL_CLASSES = ['crazing', 'rolled-in_scale', 'scratches', 'crease']

# --- CUSTOM CSS ---
st.markdown('''
<style>
    @keyframes fadeIn { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }
    .stApp { background-color: #0e1117; }
    h1 { font-family: 'Inter', sans-serif; font-weight: 700; letter-spacing: -1px; }
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
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
''', unsafe_allow_html=True)

# --- LOAD MODELS ---
@st.cache_resource
def load_models():
    model_url = 'https://github.com/maashg3l4/aegisvision/releases/download/v3.0.0/best.pt'
    policy_url = 'https://raw.githubusercontent.com/maashg3l4/aegisvision/main/src/policy/routing_policy_v3.json'
    if not os.path.exists('/tmp/best.pt'):
        urllib.request.urlretrieve(model_url, '/tmp/best.pt')
    if not os.path.exists('/tmp/policy.json'):
        urllib.request.urlretrieve(policy_url, '/tmp/policy.json')
    model = YOLO('/tmp/best.pt')
    with open('/tmp/policy.json') as f:
        policy = json.load(f)
    return model, policy

model, policy = load_models()
kb = load_knowledge_base()

try:
    fm = FlywheelManager()
    flywheel_available = True
except Exception as e:
    flywheel_available = False

# --- FEATURE EXTRACTION ---
FEATURES = ['max_conf','mean_conf','std_conf','p90_conf','top1_gap','conf_entropy', 'n_preds','n_mid','n_high','mean_area','min_area','mean_ar','std_ar', 'small_ratio','unique_classes_pred']

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
st.title('AegisVision: Defect Inspection Assistant')
st.markdown("<p style='color: #94a3b8; font-size: 16px;'>Upload an image for mathematical risk assessment.</p>", unsafe_allow_html=True)

groq_key = st.secrets.get('GROQ_API_KEY', '')
if not groq_key:
    with st.sidebar:
        st.header('Configuration')
        st.text_input('Groq API Key:', type='password', help='Required for AI Chat Assistant', key='groq_api_key')
    groq_key = st.session_state.get('groq_api_key', '')

if 'messages' not in st.session_state:
    st.session_state.messages = []

uploaded_file = st.file_uploader('', type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption='Original Upload', use_column_width=True)

    if st.button('Run Inspection', use_container_width=True):
        st.session_state.messages = []
        
        with st.spinner('Analyzing with YOLO11s and 16-Feature Policy...'):
            img_np = np.array(image)
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            results = model(img_bgr, conf=0.001, verbose=False)[0]

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

            # --- THE ULTIMATE VISUAL FILTER ---
            vis_results = results
            if len(results.boxes) > 0:
                boxes_to_keep = []
                for i, box in enumerate(results.boxes):
                    conf = float(box.conf[0])
                    cls_id = int(box.cls[0])
                    box_class_name = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else 'unknown'
                    
                    # If it's critical, threshold is 0.30. Otherwise, 0.40.
                    threshold = 0.30 if box_class_name in CRITICAL_CLASSES else 0.40
                    
                    if conf >= threshold:
                        boxes_to_keep.append(i)
                
                if boxes_to_keep:
                    vis_results = results[boxes_to_keep]

            if is_ood:
                annotated_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            else:
                annotated_bgr = vis_results.plot(conf=True, line_width=1)
                annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

            with col2:
                st.image(annotated_rgb, caption='Filtered Detections', use_column_width=True)

            feats_dict = extract_features(results)
            x = np.array([feats_dict[f] for f in FEATURES])
            mu = np.array(policy['scaler']['mean'])[:15]
            sd = np.array(policy['scaler']['scale'])[:15]
            w = np.array(policy['lr_model']['coefficients'])[:15]
            b = policy['lr_model']['intercept']
            z = float(((x - mu) / sd) @ w + b)
            danger_score = 1.0 / (1.0 + np.exp(-z))

            top_class_id = int(results.boxes.cls[0].item()) if len(results.boxes) > 0 else -1
            top_class_name = CLASS_NAMES[top_class_id] if top_class_id >= 0 and top_class_id < len(CLASS_NAMES) else 'unknown'
            force_review = top_class_name in CRITICAL_CLASSES

            st.markdown('---')
            if is_ood:
                st.error(f'OUT-OF-DISTRIBUTION DETECTED (Energy: {energy_score:.2f}) - MANUAL REVIEW REQUIRED')
                decision = 'INSUFFICIENT_EVIDENCE'
            elif force_review:
                st.error(f'HUMAN REVIEW REQUIRED (Critical Class: {top_class_name.upper()} detected) - Safety Override')
                decision = 'REQUIRES_INSPECTION'
            else:
                if danger_score > 0.80:
                    st.error(f'HUMAN REVIEW REQUIRED (Risk Score: {danger_score:.4f})')
                    decision = 'REQUIRES_INSPECTION'
                else:
                    st.success(f'AUTO-ACCEPTED (Risk Score: {danger_score:.4f})')
                    decision = 'AUTO_ASSESSED'

            st.session_state['inspection_record'] = {
                'inspection_id': 'live_inspection',
                'detections': [
                    {'class_name': CLASS_NAMES[int(box.cls[0])], 'confidence': float(box.conf[0])}
                    for box in results.boxes
                ] if len(results.boxes) > 0 else [],
                'uncertainty_assessment': {
                    'risk_score': float(danger_score),
                    'is_ood': bool(is_ood),
                    'safety_override_triggered': bool(force_review)
                },
                'routing_decision': decision
            }

            if not is_ood and len(results.boxes) > 0:
                try:
                    report_text = generate_report(top_class_name, float(results.boxes.conf[0].item()), danger_score, kb, force_review)
                    lines = report_text.strip().split('\n')
                    html = "<div class='report-card'><div class='report-title'>Aegis Report Assistant</div><div class='report-text'>"
                    for line in lines:
                        if 'RECOMMENDED ACTION' in line or 'Inspect' in line:
                            html += f"<span class='action-item'>{line.strip()}</span>"
                        elif '===' in line:
                            continue
                        else:
                            html += f"{line.strip()}<br>"
                    html += '</div></div>'
                    st.markdown(html, unsafe_allow_html=True)
                except Exception as e:
                    st.warning(f'Report Assistant error: {e}')

            st.markdown('---')
            st.subheader('Inspector Feedback')
            st.caption('Your feedback trains the next version of the model.')
            
            temp_img_path = '/tmp/temp_upload.jpg'
            cv2.imwrite(temp_img_path, img_bgr)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button('✅ Correct', use_container_width=True):
                    if flywheel_available:
                        fm.save_feedback(temp_img_path, top_class_name, float(results.boxes.conf[0].item()) if len(results.boxes) > 0 else 0.0, danger_score, 'CORRECT')
                        st.success('Feedback saved to Drive/SQLite. Thank you!')
                    else:
                        st.info('Feedback received! (Demo Mode)')
            with col2:
                if st.button('❌ False Positive', use_container_width=True):
                    if flywheel_available:
                        fm.save_feedback(temp_img_path, top_class_name, float(results.boxes.conf[0].item()) if len(results.boxes) > 0 else 0.0, danger_score, 'FALSE_POSITIVE')
                        st.success('Feedback saved to Drive/SQLite. Thank you!')
                    else:
                        st.info('Feedback received! (Demo Mode)')
            with col3:
                if st.button('⚠️ Missed Defect', use_container_width=True):
                    if flywheel_available:
                        fm.save_feedback(temp_img_path, top_class_name, float(results.boxes.conf[0].item()) if len(results.boxes) > 0 else 0.0, danger_score, 'MISSED_DEFECT')
                        st.success('Feedback saved to Drive/SQLite. Thank you!')
                    else:
                        st.info('Feedback received! (Demo Mode)')

# --- CONVERSATIONAL XAI CHAT ---
if 'inspection_record' in st.session_state:
    st.markdown('---')
    st.subheader('Ask Aegis')

    for message in st.session_state.messages:
        with st.chat_message(message['role']):
            st.markdown(message['content'])

    if not groq_key:
        st.info('Please enter a Groq API Key in the sidebar to enable the AI Chat Assistant.')
    else:
        st.markdown('**Suggested Questions:**')
        col1, col2, col3 = st.columns(3)
        suggested_q = None
        with col1:
            if st.button('What defect was found?', use_container_width=True):
                suggested_q = 'What defect was found?'
        with col2:
            if st.button('Why is this flagged?', use_container_width=True):
                suggested_q = 'Why is this image sent for human review?'
        with col3:
            if st.button('What is the risk score?', use_container_width=True):
                suggested_q = 'What is the risk score of this inspection?'
                
        user_question = suggested_q 
        chat_input_val = st.chat_input('Ask about this inspection...')
        if chat_input_val:
            user_question = chat_input_val

        if user_question:
            st.session_state.messages.append({'role': 'user', 'content': user_question})
            with st.chat_message('user'):
                st.markdown(user_question)

            with st.chat_message('assistant'):
                with st.spinner('Aegis is thinking...'):
                    try:
                        client = Groq(api_key=groq_key)
                        prompt = build_prompt(st.session_state['inspection_record'], user_question)
                        chat_completion = client.chat.completions.create(
                            messages=[
                                {'role': 'system', 'content': SYSTEM_PROMPT},
                                {'role': 'user', 'content': prompt}
                            ],
                            model='openai/gpt-oss-20b',
                            temperature=0.0,
                            response_format={'type': 'json_object'}
                        )
                        raw_res = chat_completion.choices[0].message.content
                        res_json = json.loads(raw_res)
                        schema_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'configs', 'schemas', 'chat_response.schema.json')
                        with open(schema_path, 'r') as f:
                            schema = json.load(f)
                        validate(instance=res_json, schema=schema)

                        if res_json['is_abstained']:
                            final_answer = f"⚠️ **Aegis Abstained**: {res_json['abstain_reason']}"
                        else:
                            final_answer = res_json['answer_text']
                            if res_json['evidence_sources']:
                                final_answer += f"\n\n*Sources: {', '.join(res_json['evidence_sources'])}*"
                        
                        st.markdown(final_answer)
                        st.session_state.messages.append({'role': 'assistant', 'content': final_answer})

                    except ValidationError:
                        err = '🛑 **Guardrail Blocked**: The AI hallucinated a response. This was rejected.'
                        st.error(err)
                        st.session_state.messages.append({'role': 'assistant', 'content': err})
                    except Exception as e:
                        err = f'❌ **Chat Error**: {e}'
                        st.error(err)
                        st.session_state.messages.append({'role': 'assistant', 'content': err})
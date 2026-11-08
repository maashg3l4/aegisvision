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
from src.report_assistant.vlm_prompt import SYSTEM_PROMPT, build_prompt, friendly_chat_error
from src.flywheel.manager import FlywheelManager
from src.policy.routing_policy import filter_detection_indices, routing_decision

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
    fm = FlywheelManager(
        supabase_url=st.secrets.get('SUPABASE_URL', os.environ.get('SUPABASE_URL')),
        supabase_key=st.secrets.get('SUPABASE_KEY', os.environ.get('SUPABASE_KEY')),
    )
    flywheel_available = True
    flywheel_error = None
except Exception as e:
    flywheel_available = False
    flywheel_error = str(e)

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

# --- STATE MANAGEMENT ---
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'inspection_results' not in st.session_state:
    st.session_state.inspection_results = None

# --- UI ---
st.title('AegisVision: Defect Inspection Assistant')
st.markdown("<p style='color: #94a3b8; font-size: 16px;'>Upload an image for mathematical risk assessment.</p>", unsafe_allow_html=True)

groq_key = st.secrets.get('GROQ_API_KEY', '')
if not groq_key:
    with st.sidebar:
        st.header('Configuration')
        st.text_input('Groq API Key:', type='password', help='Required for AI Chat Assistant', key='groq_api_key')
    groq_key = st.session_state.get('groq_api_key', '')

uploaded_file = st.file_uploader('', type=['jpg', 'jpeg', 'png'])

if st.button('Run Inspection', use_container_width=True) and uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    img_np = np.array(image)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    
    with st.spinner('Analyzing...'):
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

        detection_confidences = results.boxes.conf.cpu().numpy().tolist()
        detection_classes = results.boxes.cls.cpu().numpy().astype(int).tolist()
        boxes_to_keep = filter_detection_indices(
            detection_confidences, detection_classes, CLASS_NAMES, CRITICAL_CLASSES
        )
        vis_results = results[boxes_to_keep]

        if is_ood:
            annotated_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        else:
            annotated_bgr = vis_results.plot(conf=True, line_width=1)
            annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

        feats_dict = extract_features(results)
        x = np.array([feats_dict[f] for f in FEATURES])
        mu = np.array(policy['scaler']['mean'])[:15]
        sd = np.array(policy['scaler']['scale'])[:15]
        w = np.array(policy['lr_model']['coefficients'])[:15]
        b = policy['lr_model']['intercept']
        z = float(((x - mu) / sd) @ w + b)
        danger_score = 1.0 / (1.0 + np.exp(-z))

        top_class_id = int(vis_results.boxes.cls[0].item()) if len(vis_results.boxes) > 0 else -1
        top_class_name = CLASS_NAMES[top_class_id] if top_class_id >= 0 and top_class_id < len(CLASS_NAMES) else 'unknown'
        force_review = top_class_name in CRITICAL_CLASSES
        decision = routing_decision(
            danger_score,
            force_review=force_review,
            is_ood=is_ood,
            has_detection=len(vis_results.boxes) > 0,
        )

        # Save to Session State
        st.session_state.inspection_results = {
            'image': image,
            'results': results,
            'is_ood': is_ood,
            'energy_score': energy_score,
            'danger_score': danger_score,
            'top_class_name': top_class_name,
            'top_confidence': float(vis_results.boxes.conf[0].item()) if len(vis_results.boxes) > 0 else 0.0,
            'force_review': force_review,
            'decision': decision,
            'visible_results': vis_results,
            'annotated_rgb': annotated_rgb
        }
        st.session_state.messages = []

# --- RENDER RESULTS (FROM STATE) ---
if st.session_state.inspection_results is not None:
    res = st.session_state.inspection_results
    
    col1, col2 = st.columns(2)
    with col1:
        st.image(res['image'], caption='Original Upload', use_column_width=True)
    with col2:
        st.image(res['annotated_rgb'], caption='Filtered Detections', use_column_width=True)

    st.markdown('---')
    if res['is_ood']:
        st.error(f'OUT-OF-DISTRIBUTION DETECTED (Energy: {res["energy_score"]:.2f}) - MANUAL REVIEW REQUIRED')
    elif res['force_review']:
        st.error(f'HUMAN REVIEW REQUIRED (Critical Class: {res["top_class_name"].upper()} detected) - Safety Override')
    elif res['decision'] == 'REQUIRES_INSPECTION':
        reason = 'No detections met the display confidence threshold' if len(res['visible_results'].boxes) == 0 else 'Risk score exceeds the review threshold'
        st.error(f'HUMAN REVIEW REQUIRED ({reason}; Risk Score: {res["danger_score"]:.4f})')
    else:
        st.success(f'AUTO-ACCEPTED (Risk Score: {res["danger_score"]:.4f})')

    if not res['is_ood'] and len(res['visible_results'].boxes) > 0:
        try:
            report_text = generate_report(res['top_class_name'], res['top_confidence'], res['danger_score'], kb, res['force_review'])
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
    temp_img_path = '/tmp/temp_upload.jpg'
    temp_img_ready = cv2.imwrite(
        temp_img_path, cv2.cvtColor(np.array(res['image']), cv2.COLOR_RGB2BGR)
    )
    if not temp_img_ready:
        st.error('Feedback was not saved: could not prepare the uploaded image.')

    def save_feedback(feedback_type):
        if not temp_img_ready:
            return
        if not flywheel_available:
            st.error(f'Feedback was not saved because flywheel storage is unavailable: {flywheel_error}')
            return
        try:
            object_path = fm.save_feedback(
                temp_img_path,
                res['top_class_name'],
                res['top_confidence'],
                res['danger_score'],
                feedback_type,
            )
            st.success(
                f'Image uploaded to Supabase Storage: defect-images/{object_path}. '
                'Metadata was saved to this app’s SQLite database.'
            )
        except Exception as e:
            st.error(f'Feedback save failed: {e}')

    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button('✅ Correct', use_container_width=True):
            save_feedback('CORRECT')
    with col2:
        if st.button('❌ False Positive', use_container_width=True):
            save_feedback('FALSE_POSITIVE')
    with col3:
        if st.button('⚠️ Missed Defect', use_container_width=True):
            save_feedback('MISSED_DEFECT')

# --- CHAT ---
if st.session_state.inspection_results is not None:
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
                        current_result = st.session_state.inspection_results
                        inspection_record = {
                            'inspection_id': 'live',
                            'detections': [
                                {
                                    'class_name': CLASS_NAMES[int(box.cls[0])],
                                    'confidence': float(box.conf[0]),
                                }
                                for box in current_result['visible_results'].boxes
                            ],
                            'total_detections': len(current_result['visible_results'].boxes),
                            'uncertainty_assessment': {
                                'risk_score': current_result['danger_score'],
                                'is_ood': current_result['is_ood'],
                                'safety_override_triggered': current_result['force_review'],
                            },
                            'routing_decision': current_result['decision'],
                        }
                        prompt = build_prompt(inspection_record, user_question)
                        chat_completion = client.chat.completions.create(
                            messages=[{'role': 'system', 'content': SYSTEM_PROMPT}, {'role': 'user', 'content': prompt}],
                            model='openai/gpt-oss-20b',
                            temperature=0.0,
                            max_completion_tokens=512,
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
                    except Exception as e:
                        st.error(friendly_chat_error(e))
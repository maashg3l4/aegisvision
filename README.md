<div align="center"> <a href="https://git.io/typing-svg"> <img src="https://readme-typing-svg.herokuapp.com?font=Fira+Code&weight=600&size=28&pause=1000&color=3B82F6&center=true&vCenter=true&width=850&lines=AegisVision+V4.2;17-Class+Safety-First+Inspection;Fail-Closed+Routing+%7C+OOD+Shield+%7C+Grounded+LLM" alt="Typing SVG" /> </a> <p><strong>A safety-first, human-in-the-loop routing framework for industrial steel surface defect detection.</strong></p> <p>Built with PyTorch, YOLO26s, Streamlit, FastAPI, and deterministic LLM grounding.</p> <!-- Badges -->

<a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.10+-blue.svg?logo=python&logoColor=white" alt="Python"></a> <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch"></a> <a href="https://streamlit.io/"><img src="https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white" alt="Streamlit"></a> <a href="https://fastapi.tiangolo.com/"><img src="https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white" alt="FastAPI"></a> <a href="https://github.com/maashg3l4/aegisvision/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License"></a>

<br><br>

🚀 <a href="https://aegisvision-n84itsqjrquzfpkwu5x3d9.streamlit.app/">Click Here to Try the Live Interactive Demo</a>
</div>
🎯 The Core Problem: Silent AI Failures

In industrial manufacturing, AI models fail silently. A YOLO model might confidently hallucinate a bounding box on a pristine surface, or miss a critical micro-crack. Furthermore, when a model is shown an Out-of-Distribution (OOD) image like a painted surface instead of raw steel it will confidently guess.

Inspectors cannot trust a black-box AI. AegisVision solves this by answering one question: "When should a human trust the AI, and when must they be forced to review?"

🧠 System Architecture (V4.2)
The "Eye" (Object Detection): A YOLO26s model trained on a 17-Class Unified Dataset (merging NEU-DET, GC10-DET, and CR7-DET) using Class Weighting (cls_pw=1.0) to penalize missing rare defects.
The "Brain" (Fail-Closed Routing): A 16-feature Logistic Regression router. Crucially, it defaults to HUMAN_REVIEW unless the image is explicitly proven safe.
The "Shield" (Dual OOD Detection): A Hard Color Check (HSV saturation) combined with an Energy-based Out-of-Distribution score. If an image is from a different domain, it halts inference and triggers a manual review.
The "Voice" (Deterministic RAG): A strict JSON-schema LLM pipeline. It prepends a deterministic status block (e.g., Decision: HUMAN_REVIEW. Reasons: OUT_OF_DOMAIN) before generating any text, preventing the LLM from hallucinating the system's state.
🏗 Architecture at a Glance

The pipeline is organised as four layers, each with a single responsibility. Data flows left to right; any layer can escalate to a human, and no layer can silently approve on behalf of another.

in-domain
verified safe
OUT_OF_DOMAIN
low conf, rolled_pit,uncertain
retraining data
📷 Steel Image
🛡 L1 · OOD ShieldHSV saturation + energy
👁 L2 · YOLO26s17-class detector
🧠 L3 · Safety Router16-feature LogReg + rules
💬 L4 · Grounded LLMstatus block prepended
✅ AUTO_PASSexplicitly proven safe
🛑 HUMAN_REVIEWdefault · fail-closed
🔄 Supabase Flywheelhuman corrections

🔴 Fast edges are fail-closed escalations to a human. 🟢 AUTO_PASS is the only path to automation.

Layer responsibilities
Layer	Component	Responsibility	Failure behaviour
L1 · Shield	HSV saturation check + energy-based OOD score	Decide whether the image belongs to the grayscale steel domain at all	Halts inference → HUMAN_REVIEW (OUT_OF_DOMAIN)
L2 · Eye	YOLO26s, 17 unified classes, class-weighted (cls_pw=1.0)	Localise and classify defects	Low-confidence boxes are passed on, never hidden
L3 · Brain	16-feature Logistic Regression + deterministic safety overrides	Convert detector output into a routing decision	Defaults to HUMAN_REVIEW unless explicitly proven safe
L4 · Voice	Strict JSON-schema LLM pipeline	Explain the decision in natural language	Status block is injected before generation
End-to-end sequence
Supabase Flywheel
L4 · Grounded LLM
L3 · Safety Router
L2 · YOLO26s
L1 · OOD Shield
Streamlit UI
Supabase Flywheel
L4 · Grounded LLM
L3 · Safety Router
L2 · YOLO26s
L1 · OOD Shield
Streamlit UI
Inference halted, no boxes drawn
alt
[Out-of-domain]
[In-domain]
Inspector
Upload steel image
1
HSV saturation + energy score
2
OUT_OF_DOMAIN
3
Pass image
4
Boxes, classes, confidences
5
16 features → LogReg → safety overrides
6
Decision + reasons (JSON)
7
Prepend deterministic status block
8
Grounded explanation
9
Decision, boxes, explanation
10
Correction (if any)
11
Future retraining data
12
Inspector
Router state machine
default state (fail-closed)
image received
OUT_OF_DOMAIN
in-domain
boxes + features
conf < 0.60
rolled_pit detected
no override fired
router proves safe
otherwise
HUMAN_REVIEW
OOD_CHECK
DETECT
SAFETY_RULES
CLASSIFY
AUTO_PASS
Design principles
Fail-closed by construction. HUMAN_REVIEW is the initial state; only a positive proof moves an image out of it.
Defence in depth. The OOD shield, the learned router and hard-coded safety overrides are independent. One layer's mistake does not become another's approval.
Determinism where it matters. The learned model proposes; deterministic rules and a templated status block dispose.
Auditable language. The LLM explains decisions but cannot alter them.
🏆 Verified Performance & Engineering Metrics
Metric	Result	Engineering Impact
17-Class Generalist mAP50	0.767	Strong multi-domain generalization (YOLO26s).
Class Imbalance Mitigation	crease: 0.437	Class weighting successfully forced the model to learn rare defects.
Safety Routing (Fail-Closed)	100%	Bounding boxes <0.60 confidence are explicitly flagged for review.
OOD Detection (Energy + Color)	PASS	The teal synthetic image is successfully rejected as OUT_OF_DOMAIN.
LLM Grounding	100%	Deterministic prefix injection prevents the LLM from contradicting the router.
📈 Visual Evidence
1. The Shield: OOD Detection in Action

When an image falls outside the grayscale steel domain (e.g., a teal painted surface), the router refuses to draw bounding boxes and forces a manual review, overriding the model's low-confidence hallucinations. OOD Synthetic Test (Placeholder - replace with your new screenshot)

2. The Routing Logic

The chatbot uses deterministic JSON grounding. It cannot invent thresholds or claim an image was "sent for review" if it was actually rejected.

🛑 Brutal Honesty: Failure Analysis & Known Limitations

A production-grade system must document where it fails.

The rolled_pit Failure Mode: Despite class weighting and mixup augmentations, the rolled_pit class remains at 0.228 mAP50. Why? The validation set only contains 9 instances of this defect. A model cannot learn the true visual distribution of a defect from 9 images. Mitigation: The router forces every rolled_pit detection into HUMAN_REVIEW via a Safety Override.
Supabase Flywheel Latency: The feedback loop (capturing human corrections) is implemented via Supabase. However, due to an IPv4/IPv6 routing incompatibility between Streamlit Community Cloud and Supabase's default direct connection, the image upload currently fails. Mitigation: We are actively migrating to the Supabase IPv4 Session Pooler.
Monolithic UI vs API: Currently, the Streamlit frontend runs YOLO locally. A FastAPI backend has been drafted to decouple this architecture, but the live demo still uses the local inference for stability.
⚙️ Reproducibility & Engineering

We follow strict MLOps protocols. The training pipeline uses an Atomic Auto-Save Callback that prevents Google Drive FUSE corruption by verifying file sizes and performing atomic replacements.

Reproduce Locally
bash
git clone https://github.com/maashg3l4/aegisvision.git
cd aegisvision
pip install -r requirements.txt

# Run the Streamlit App
streamlit run src/frontend/app.py

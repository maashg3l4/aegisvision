<div align="center"> <img src="https://capsule-render.vercel.app/api?type=waving&color=0:0b1220,50:1e3a8a,100:3B82F6&height=210&section=header&text=AegisVision%20V4.2&fontSize=60&fontColor=ffffff&fontAlignY=38&desc=Safety-First%20%E2%80%A2%20Human-in-the-Loop%20%E2%80%A2%20Fail-Closed&descSize=18&descAlignY=60&animation=fadeIn" width="100%" alt="AegisVision banner"/> <a href="https://git.io/typing-svg"> <img src="https://readme-typing-svg.herokuapp.com?font=Fira+Code&weight=600&size=24&pause=1200&color=3B82F6&center=true&vCenter=true&width=900&lines=When+should+a+human+trust+the+AI%3F;17-Class+Safety-First+Inspection;Fail-Closed+Routing+%E2%80%A2+Dual+OOD+Shield+%E2%80%A2+Grounded+LLM;Industrial+Steel+Surface+Defect+Detection" alt="Typing SVG" /> </a> <br/>

A safety-first, human-in-the-loop routing framework for industrial steel surface defect detection. <br/> <sub>Built with PyTorch, YOLO26s, Streamlit, FastAPI, and deterministic LLM grounding.</sub>

<br/><br/>

<a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"></a> <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch"></a> <a href="https://streamlit.io/"><img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit"></a> <a href="https://fastapi.tiangolo.com/"><img src="https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white" alt="FastAPI"></a> <a href="https://github.com/maashg3l4/aegisvision/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-MIT-22C55E?style=for-the-badge" alt="License"></a>

<br/> <img src="https://img.shields.io/badge/mAP50-0.767-3B82F6?style=flat-square" alt="mAP50"/> <img src="https://img.shields.io/badge/Classes-17-8B5CF6?style=flat-square" alt="Classes"/> <img src="https://img.shields.io/badge/Router%20Features-16-F59E0B?style=flat-square" alt="Features"/> <img src="https://img.shields.io/badge/Confidence%20Floor-0.60-EF4444?style=flat-square" alt="Confidence floor"/> <img src="https://img.shields.io/badge/Default-HUMAN__REVIEW-DC2626?style=flat-square" alt="Default"/>

<br/><br/>

<a href="https://aegisvision-n84itsqjrquzfpkwu5x3d9.streamlit.app/"><img src="https://img.shields.io/badge/🚀%20CLICK%20HERE%20TO%20TRY%20THE%20LIVE%20INTERACTIVE%20DEMO-FF4B4B?style=for-the-badge" alt="Live Demo"/></a>

<br/><br/>

🎯 Problem · 🏗 Architecture · 🏆 Results · 📈 Evidence · 🛑 Limitations · ⚙️ Reproduce

</div>
🎯 The Core Problem: Silent AI Failures

In industrial manufacturing, AI models fail silently. A YOLO model might confidently hallucinate a bounding box on a pristine surface, or miss a critical micro-crack. Furthermore, when a model is shown an Out-of-Distribution (OOD) image like a painted surface instead of raw steel, it will confidently guess.

Inspectors cannot trust a black-box AI. AegisVision solves this by answering one question:

"When should a human trust the AI, and when must they be forced to review?"

The design principle is fail-closed: the default outcome for every image is HUMAN_REVIEW. Automation is something an image must earn.

<div align="center">
🚫 Typical pipeline	✅ AegisVision
Trusts every detection	Trusts nothing until proven safe
Guesses on unfamiliar images	Rejects out-of-domain images before detection
LLM free-styles explanations	LLM is bound to a deterministic status block
Failures are silent	Failures are explicit, reasoned escalations
</div>
🧠 System Architecture (V4.2)

Four layers, each with a single responsibility. Data flows left to right; any layer can escalate to a human, and no layer can silently approve on behalf of another.

🔭 Live Pipeline View
in-domain
verified safe
OUT_OF_DOMAIN
low conf · rolled_pit ·uncertain
retraining data
📷 Steel Image
🛡 THE SHIELD · L1HSV saturation + energyOOD
👁 THE EYE · L2YOLO26s · 17 classes
🧠 THE BRAIN · L316-feature LogReg + safetyrules
💬 THE VOICE · L4Grounded LLM · JSONschema
✅ AUTO_PASSexplicitly proven safe
🛑 HUMAN_REVIEWdefault · fail-closed
🔄 Supabase Flywheelhuman corrections

🔴 Fast-flowing edges = fail-closed escalations to a human. 🟢 AUTO_PASS is the only path to automation.

🧩 The Four Layers
<table> <tr> <td width="25%" valign="top">
🛡 L1 · The Shield

Dual OOD Detection

A Hard Color Check (HSV saturation) combined with an Energy-based Out-of-Distribution score. If an image is from a different domain, it halts inference and triggers a manual review.

</td> <td width="25%" valign="top">
👁 L2 · The Eye

Object Detection

A YOLO26s model trained on a 17-Class Unified Dataset (merging NEU-DET, GC10-DET, and CR7-DET) using Class Weighting (cls_pw=1.0) to penalize missing rare defects.

</td> <td width="25%" valign="top">
🧠 L3 · The Brain

Fail-Closed Routing

A 16-feature Logistic Regression router. Crucially, it defaults to HUMAN_REVIEW unless the image is explicitly proven safe.

</td> <td width="25%" valign="top">
💬 L4 · The Voice

Deterministic RAG

A strict JSON-schema LLM pipeline. It prepends a deterministic status block (e.g., Decision: HUMAN_REVIEW. Reasons: OUT_OF_DOMAIN) before generating any text, preventing the LLM from hallucinating the system's state.

</td> </tr> </table>
📊 Layer Contract
Layer	Component	Responsibility	Failure behaviour
L1 · Shield	HSV saturation + energy OOD score	Decide whether the image belongs to the grayscale steel domain	Halts inference → HUMAN_REVIEW (OUT_OF_DOMAIN)
L2 · Eye	YOLO26s, class-weighted (cls_pw=1.0)	Localise and classify defects	Low-confidence boxes are passed on, never hidden
L3 · Brain	16-feature LogReg + deterministic overrides	Turn detector output into a routing decision	Defaults to HUMAN_REVIEW unless proven safe
L4 · Voice	JSON-schema LLM	Explain the decision in natural language	Status block is injected before generation
🔁 End-to-End Sequence
🔄 Supabase Flywheel
💬 L4 · Grounded LLM
🧠 L3 · Safety Router
👁 L2 · YOLO26s
🛡 L1 · OOD Shield
Streamlit UI
🔄 Supabase Flywheel
💬 L4 · Grounded LLM
🧠 L3 · Safety Router
👁 L2 · YOLO26s
🛡 L1 · OOD Shield
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
🚦 Router State Machine
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
🧭 Design Principles
Fail-closed by construction. HUMAN_REVIEW is the initial state; only a positive proof moves an image out of it.
Defence in depth. The OOD shield, the learned router and hard-coded safety overrides are independent. One layer's mistake never becomes another's approval.
Determinism where it matters. The learned model proposes; deterministic rules and a templated status block dispose.
Auditable language. The LLM explains decisions but cannot alter them.
🏆 Verified Performance & Engineering Metrics
<div align="center">
Metric	Result	Engineering Impact
17-Class Generalist mAP50	0.767	Strong multi-domain generalization (YOLO26s).
Class Imbalance Mitigation	crease: 0.437	Class weighting successfully forced the model to learn rare defects.
Safety Routing (Fail-Closed)	100%	Bounding boxes < 0.60 confidence are explicitly flagged for review.
OOD Detection (Energy + Color)	PASS	The teal synthetic image is successfully rejected as OUT_OF_DOMAIN.
LLM Grounding	100%	Deterministic prefix injection prevents the LLM from contradicting the router.
</div>
📈 Visual Evidence
1 · The Shield: OOD Detection in Action

When an image falls outside the grayscale steel domain (e.g., a teal painted surface), the router refuses to draw bounding boxes and forces a manual review, overriding the model's low-confidence hallucinations.

<!-- Add your screenshot once it is committed to the repo, then remove these comment markers: <div align="center"> <img src="assets/ood_energy_distribution.png" alt="OOD Synthetic Test" width="80%"/> </div> -->
2 · The Routing Logic

The chatbot uses deterministic JSON grounding. It cannot invent thresholds or claim an image was "sent for review" if it was actually rejected.

text
Decision: HUMAN_REVIEW. Reasons: OUT_OF_DOMAIN
🛑 Brutal Honesty: Failure Analysis & Known Limitations

A production-grade system must document where it fails.

<details open> <summary><b>1 · The <code>rolled_pit</code> Failure Mode</b></summary> <br/>

Despite class weighting and mixup augmentations, the rolled_pit class remains at 0.228 mAP50.

Why? The validation set only contains 9 instances of this defect. A model cannot learn the true visual distribution of a defect from 9 images.
Mitigation: The router forces every rolled_pit detection into HUMAN_REVIEW via a Safety Override.
</details> <details open> <summary><b>2 · Supabase Flywheel Latency</b></summary> <br/>

The feedback loop (capturing human corrections) is implemented via Supabase. However, due to an IPv4/IPv6 routing incompatibility between Streamlit Community Cloud and Supabase's default direct connection, the image upload currently fails.

Mitigation: We are actively migrating to the Supabase IPv4 Session Pooler.
</details> <details open> <summary><b>3 · Monolithic UI vs API</b></summary> <br/>

Currently, the Streamlit frontend runs YOLO locally. A FastAPI backend has been drafted to decouple this architecture, but the live demo still uses local inference for stability.

</details>
🗺 Roadmap
[x] 17-class unified detector
[x] Fail-closed router with safety overrides
[x] Dual OOD shield (HSV + energy)
[x] Deterministic LLM grounding
[ ] Supabase IPv4 Session Pooler migration
[ ] Decouple inference behind FastAPI
[ ] Collect more rolled_pit samples and re-evaluate
[ ] Router calibration and reliability curves
⚙️ Reproducibility & Engineering

We follow strict MLOps protocols. The training pipeline uses an Atomic Auto-Save Callback that prevents Google Drive FUSE corruption by verifying file sizes and performing atomic replacements.

Reproduce Locally
bash
git clone https://github.com/maashg3l4/aegisvision.git
cd aegisvision
pip install -r requirements.txt

# Run the Streamlit App
streamlit run src/frontend/app.py
<div align="center">

Built on one belief: a model that knows when it doesn't know is worth more than one that is merely accurate.

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:3B82F6,50:1e3a8a,100:0b1220&height=120&section=footer" width="100%" alt="footer"/> </div>

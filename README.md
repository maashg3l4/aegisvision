<div align="center"> <img src="https://capsule-render.vercel.app/api?type=waving&color=0:0b1220,100:3b82f6&height=180&section=header&text=AegisVision&fontColor=ffffff&fontSize=56&fontAlignY=38&animation=fadeIn&desc=Safety-first%20routing%20for%20industrial%20steel%20defect%20inspection&descSize=16&descAlignY=60" alt="AegisVision" width="100%"/> <a href="https://git.io/typing-svg"> <img src="https://readme-typing-svg.herokuapp.com?font=Fira+Code&weight=600&size=22&pause=1200&color=3B82F6&center=true&vCenter=true&width=820&lines=Fail-closed+by+design;17-class+steel+surface+defect+detection;Dual+OOD+shield+%7C+Human-in-the-loop+routing;Deterministically+grounded+LLM+explanations" alt="AegisVision tagline" /> </a> <br/>

Python PyTorch Streamlit FastAPI License: MIT

Live Demo  ·  Architecture  ·  Results  ·  Limitations  ·  Quickstart

</div>
Abstract

Industrial vision models fail silently. A detector can place a confident box on a pristine surface, miss a micro-crack, or, when shown an image from a different domain, answer with the same confidence as it would on familiar data. For an inspector, the accuracy of the model matters less than a different question:

When can a human trust the AI, and when must they be forced to look?

AegisVision treats this as a routing problem rather than a detection problem. A 17-class YOLO26s detector produces candidate defects, and a fail-closed router decides whether the result may pass automatically. The router defaults to HUMAN_REVIEW and releases an image only when it is positively shown to be in-domain, confident, and free of safety-override classes. A grounded explanation layer then reports that decision without being able to contradict it.

Contributions
	
Fail-closed routing	A 16-feature logistic-regression router whose default outcome is human review. Automation must be earned per image.
Dual OOD shield	A hard HSV-saturation check combined with an energy-based OOD score halts inference on foreign-domain inputs before any box is drawn.
Unified 17-class dataset	NEU-DET, GC10-DET and CR7-DET merged into one label space, trained with class weighting to counter rare-defect imbalance.
Deterministic grounding	The system status is injected as a fixed block ahead of LLM output under a strict JSON schema, so the language model cannot restate the system state.
Documented failure modes	Known weaknesses are listed with their mitigations. See Limitations.
Architecture
<div align="center"> <img src="https://raw.githubusercontent.com/maashg3l4/aegisvision/main/assets/architecture.svg" alt="Animated AegisVision V4.2 architecture diagram" width="100%"/> </div> <br/>

Blue packets are the nominal path. Amber packets are the OOD short-circuit: an out-of-domain image never reaches the detector, and the system emits HUMAN_REVIEW with the reason OUT_OF_DOMAIN.

System at a glance
AUTO_PASS
HUMAN_REVIEW
in-domain
out-of-domain
02 Shield
HSV saturation gate
Energy OOD score
Steel image
03 EyeYOLO26s, 17 classes
Inference haltedno boxes drawn
04 Brain16-feature LogReg routerdefault: HUMAN_REVIEW
05 Decision
06 Voicestatus block + groundedLLM
07 Inspector UI
08 FlywheelSupabase
Component summary
#	Stage	Role	Implementation
01	Input	Raw steel-surface image	Upload via Streamlit
02	Shield	Reject foreign-domain images before inference	HSV saturation gate + energy-based OOD score
03	Eye	Localise and classify defects	YOLO26s, 17 classes, cls_pw=1.0
04	Brain	Decide whether the result may skip human review	16-feature logistic regression, default HUMAN_REVIEW
05	Decision	Verdict plus machine-readable reasons	AUTO_PASS or HUMAN_REVIEW with reason codes
06	Voice	Explain the verdict in natural language	JSON-schema LLM with deterministic status prefix
07	Inspector	Final human decision	Streamlit interface
08	Flywheel	Capture human corrections for retraining	Supabase (migration in progress)
Routing logic

The decision is an ordered cascade. Each gate can only move an image toward human review. No later stage can undo an earlier rejection.

no
yes
no
yes
yes
no
no
yes
Input image
HSV saturationwithin steel range?
HUMAN_REVIEWOUT_OF_DOMAIN
Energy OOD scorebelow threshold?
YOLO26s inference
Safety override?rolled_pit detected orbox confidence below 0.60
HUMAN_REVIEWSAFETY_OVERRIDE
16-feature logisticregression router
Proven safe?
HUMAN_REVIEWLOW_ROUTER_CONFIDENCE
AUTO_PASS

Thresholds and reason-code names above are illustrative. Keep them in sync with src/ when you change them.

Grounded explanation layer

Language models describe state fluently and incorrectly. AegisVision removes that failure mode by making the status block a program output, not a model output.

Inspector UI
LLM (JSON schema)
Status Builder
Router
Inspector UI
LLM (JSON schema)
Status Builder
Router
The status line is rendered verbatim.
The LLM can elaborate but cannot restate or override it.
decision, reason codes, detections
1
Decision: HUMAN_REVIEW. Reasons: OUT_OF_DOMAIN
2
same facts as structured context
3
explanation conforming to schema
4
Design invariants
Fail-closed. Absence of evidence of safety is treated as unsafe.
Monotone gates. A gate can add reasons for review. It can never remove one.
No box without domain. Out-of-domain inputs never produce bounding boxes.
Status is data. The system state shown to the user is generated by code, not by the LLM.
Rare means review. Classes with insufficient validation support are always routed to a human.
Results

Evaluated on the validation split of the unified 17-class dataset.

Metric	Result	Notes
17-class mAP50	0.767	YOLO26s, multi-domain generalist
crease mAP50	0.437	Rare class; improved with class weighting
rolled_pit mAP50	0.228	Known weak class, see below
Low-confidence routing	Enforced	Boxes under 0.60 confidence always yield HUMAN_REVIEW (invariant by construction)
OOD rejection	Pass	Synthetic teal-surface test image rejected as OUT_OF_DOMAIN (single qualitative test)
Status grounding	Enforced	Status block is deterministic, so the LLM cannot contradict the router (invariant by construction)
Visual evidence
<div align="center"> <img src="assets/ood_energy_distribution.png" alt="Energy-score distribution and OOD rejection of a synthetic teal image" width="80%"/> <br/> <sub><b>Figure 1.</b> The shield rejects a teal painted surface before inference. No boxes are drawn and the router emits <code>HUMAN_REVIEW</code> with reason <code>OUT_OF_DOMAIN</code>.</sub> </div>
Limitations and failure analysis

A safety system is only credible if it states where it fails.

<details open> <summary><b>1. <code>rolled_pit</code> remains weak (0.228 mAP50)</b></summary> <br/>

The validation set holds only 9 instances of this class. A detector cannot learn the visual distribution of a defect from nine examples, and mAP estimated on nine samples has high variance. Class weighting and mixup did not close the gap.

Mitigation: every rolled_pit detection is forced to HUMAN_REVIEW by a safety override in the router.
Next step: collect more labelled rolled_pit data before treating this class as automatable.

</details> <details> <summary><b>2. Feedback flywheel is not yet reliable</b></summary> <br/>

Human corrections are meant to flow to Supabase for retraining. On Streamlit Community Cloud, image upload currently fails because of an IPv4/IPv6 incompatibility with Supabase's default direct connection.

Mitigation: migrating to the Supabase IPv4 Session Pooler.

</details> <details> <summary><b>3. Inference is coupled to the UI</b></summary> <br/>

The Streamlit app runs YOLO locally. A FastAPI backend has been drafted to separate inference from presentation, but the live demo still uses local inference for stability.

</details> <details> <summary><b>4. Evaluation scope</b></summary> <br/>

OOD rejection has been verified qualitatively on a synthetic example. A quantitative evaluation (AUROC and FPR at a fixed TPR against held-out foreign-domain images) is planned.

</details>
Roadmap
[x] Unified 17-class dataset and YOLO26s training
[x] Fail-closed router with safety overrides
[x] Dual OOD shield (HSV + energy)
[x] Deterministic status grounding
[ ] Supabase IPv4 Session Pooler migration
[ ] Decouple inference behind FastAPI
[ ] Quantitative OOD benchmark (AUROC, FPR@95TPR)
[ ] Router calibration analysis (reliability diagrams)
[ ] More rolled_pit and crease training data
Quickstart
bash
git clone https://github.com/maashg3l4/aegisvision.git
cd aegisvision
pip install -r requirements.txt

# Launch the Streamlit app
streamlit run src/frontend/app.py
Reproducibility

Training follows an MLOps-first protocol. The pipeline uses an atomic auto-save callback that avoids Google Drive FUSE corruption by verifying file size and performing an atomic replace before a checkpoint is considered written.

Citation
bibtex
@software{aegisvision2026,
  title  = {AegisVision: A Fail-Closed Routing Framework for Steel Surface Defect Inspection},
  author = {maashg3l4},
  year   = {2026},
  url    = {https://github.com/maashg3l4/aegisvision}
}
Acknowledgements

Built on the NEU-DET, GC10-DET and CR7-DET datasets, and on PyTorch, YOLO, Streamlit, FastAPI and Supabase. Please cite the original dataset papers when using their data.

License

Released under the MIT License.

<div align="center"> <img src="https://capsule-render.vercel.app/api?type=waving&color=0:3b82f6,100:0b1220&height=100&section=footer" width="100%" alt=""/> </div>

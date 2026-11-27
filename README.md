<div align="center">
  <a href="https://git.io/typing-svg">
    <img src="https://readme-typing-svg.herokuapp.com?font=Fira+Code&weight=600&size=28&pause=1000&color=3B82F6&center=true&vCenter=true&width=850&lines=AegisVision+V4.2;17-Class+Safety-First+Inspection;Fail-Closed+Routing+%7C+OOD+Shield+%7C+Grounded+LLM" alt="Typing SVG" />
  </a><div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:0b1220,100:3B82F6&height=190&section=header&text=AegisVision&fontSize=64&fontColor=ffffff&fontAlignY=38&desc=Safety-First%20Industrial%20Defect%20Inspection&descSize=18&descAlignY=60&animation=fadeIn" width="100%" alt="AegisVision banner"/>

<a href="https://git.io/typing-svg">
  <img src="https://readme-typing-svg.herokuapp.com?font=Fira+Code&weight=600&size=24&pause=1200&color=3B82F6&center=true&vCenter=true&width=900&lines=When+should+a+human+trust+the+AI%3F;Fail-Closed+Routing+%E2%80%A2+Dual+OOD+Shield+%E2%80%A2+Grounded+LLM;17-Class+YOLO26s+Detector+%7C+Human-in-the-Loop+by+Design" alt="Typing SVG" />
</a>

<br/>

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/License-MIT-22C55E?style=for-the-badge)](https://github.com/maashg3l4/aegisvision/blob/main/LICENSE)

<br/>

**A safety-first, human-in-the-loop routing framework for industrial steel surface defect detection.**

[**🚀 Live Demo**](https://aegisvision-n84itsqjrquzfpkwu5x3d9.streamlit.app/) · [**🏗 Architecture**](#-system-architecture) · [**📊 Results**](#-verified-performance) · [**⚠️ Limitations**](#️-failure-analysis--known-limitations) · [**⚙️ Reproduce**](#️-reproducibility)

</div>

---

## 📑 Table of Contents

1. [Problem Statement](#-problem-statement)
2. [Key Contributions](#-key-contributions)
3. [System Architecture](#-system-architecture)
4. [Routing Decision Logic](#-routing-decision-logic)
5. [Verified Performance](#-verified-performance)
6. [Visual Evidence](#-visual-evidence)
7. [Failure Analysis & Known Limitations](#️-failure-analysis--known-limitations)
8. [Roadmap](#-roadmap)
9. [Reproducibility](#️-reproducibility)

---

## 🎯 Problem Statement

Industrial vision models fail **silently**. A detector can confidently draw a box on a pristine surface, miss a micro-crack, or, when shown an **Out-of-Distribution (OOD)** image such as a painted surface instead of raw steel, still return a confident guess.

> Inspectors cannot trust a black box.
> **AegisVision answers one question: _"When should a human trust the AI, and when must they be forced to review?"_**

The design principle is **fail-closed**: the default outcome for every image is `HUMAN_REVIEW`. Automation is something an image must *earn*.

---

## ✨ Key Contributions

| # | Contribution | Why it matters |
|:-:|:--|:--|
| 1 | **17-class unified dataset** merging NEU-DET, GC10-DET and CR7-DET | Multi-domain generalisation instead of a single-benchmark model |
| 2 | **Fail-closed router** (16-feature Logistic Regression + hard safety rules) | Silent failures become explicit, auditable escalations |
| 3 | **Dual OOD shield** (HSV colour gate + energy-based score) | Rejects non-steel domains *before* any detection is trusted |
| 4 | **Deterministic LLM grounding** | The LLM cannot contradict or invent the system's state |
| 5 | **Atomic auto-save training callback** | Prevents checkpoint corruption on Google Drive FUSE |

---

## 🏗 System Architecture

The pipeline is organised as four layers, each with a single responsibility. Data flows left to right; any layer can **escalate** to a human, and no layer can silently approve on behalf of another.

```mermaid
%%{init: {"theme": "dark", "flowchart": {"curve": "basis"}}}%%
flowchart LR
    IN(["📷 Steel Image"])
    SH{{"🛡 L1 · OOD Shield<br/>HSV saturation + energy"}}
    EY["👁 L2 · YOLO26s<br/>17-class detector"]
    BR{{"🧠 L3 · Safety Router<br/>16-feature LogReg + rules"}}
    VO["💬 L4 · Grounded LLM<br/>status block prepended"]
    OK(["✅ AUTO_PASS<br/>explicitly proven safe"])
    HR(["🛑 HUMAN_REVIEW<br/>default · fail-closed"])
    FW[("🔄 Supabase Flywheel<br/>human corrections")]

    IN e1@--> SH
    SH e2@-->|in-domain| EY
    EY e3@--> BR
    BR e4@-->|verified safe| OK
    BR e5@--> VO
    SH e6@-->|OUT_OF_DOMAIN| HR
    BR e7@-->|low conf, rolled_pit, uncertain| HR
    HR e8@--> FW
    FW -.->|retraining data| EY

    e1@{ animate: true }
    e2@{ animate: true }
    e3@{ animate: true }
    e4@{ animate: true }
    e5@{ animate: true }
    e6@{ animate: true, animation: fast }
    e7@{ animate: true, animation: fast }
    e8@{ animate: true }

    classDef eye fill:#1e3a8a,stroke:#60a5fa,color:#fff
    classDef shield fill:#78350f,stroke:#fbbf24,color:#fff
    classDef brain fill:#4c1d95,stroke:#a78bfa,color:#fff
    classDef safe fill:#14532d,stroke:#4ade80,color:#fff
    classDef danger fill:#7f1d1d,stroke:#f87171,color:#fff
    classDef loop fill:#451a03,stroke:#fbbf24,color:#fff
    class IN,EY,VO eye
    class SH shield
    class BR brain
    class OK safe
    class HR danger
    class FW loop
```

> 🔴 **Red edges** are fail-closed escalations to a human. 🟢 **Green** is the only path to automation.

### Layer responsibilities

| Layer | Component | Responsibility | Failure behaviour |
|:--|:--|:--|:--|
| **L1 · Shield** | HSV saturation check + energy-based OOD score | Decide whether the image belongs to the grayscale steel domain at all | Halts inference → `HUMAN_REVIEW (OUT_OF_DOMAIN)` |
| **L2 · Eye** | YOLO26s trained on 17 unified classes, class-weighted (`cls_pw=1.0`) | Localise and classify defects | Low-confidence boxes are passed on, never hidden |
| **L3 · Brain** | 16-feature Logistic Regression + deterministic safety overrides | Convert detector output into a routing decision | Defaults to `HUMAN_REVIEW` unless explicitly proven safe |
| **L4 · Voice** | Strict JSON-schema LLM pipeline | Explain the decision in natural language | Status block is injected *before* generation |

### End-to-end sequence

```mermaid
sequenceDiagram
    autonumber
    actor Inspector
    participant UI as Streamlit UI
    participant Shield as L1 · OOD Shield
    participant Eye as L2 · YOLO26s
    participant Brain as L3 · Safety Router
    participant Voice as L4 · Grounded LLM
    participant DB as Supabase Flywheel

    Inspector->>UI: Upload steel image
    UI->>Shield: HSV saturation + energy score
    alt Out-of-domain
        Shield-->>Brain: OUT_OF_DOMAIN
        Note over Eye: Inference halted, no boxes drawn
    else In-domain
        Shield->>Eye: Pass image
        Eye->>Brain: Boxes, classes, confidences
    end
    Brain->>Brain: 16 features → LogReg → safety overrides
    Brain->>Voice: Decision + reasons (JSON)
    Voice->>Voice: Prepend deterministic status block
    Voice-->>UI: Grounded explanation
    UI-->>Inspector: Decision, boxes, explanation
    Inspector->>DB: Correction (if any)
    DB-->>Eye: Future retraining data
```

### Router state machine

```mermaid
stateDiagram-v2
    [*] --> HUMAN_REVIEW: default state (fail-closed)
    HUMAN_REVIEW --> OOD_CHECK: image received
    OOD_CHECK --> HUMAN_REVIEW: OUT_OF_DOMAIN
    OOD_CHECK --> DETECT: in-domain
    DETECT --> SAFETY_RULES: boxes + features
    SAFETY_RULES --> HUMAN_REVIEW: conf < 0.60
    SAFETY_RULES --> HUMAN_REVIEW: rolled_pit detected
    SAFETY_RULES --> CLASSIFY: no override fired
    CLASSIFY --> AUTO_PASS: router proves safe
    CLASSIFY --> HUMAN_REVIEW: otherwise
    AUTO_PASS --> [*]
    HUMAN_REVIEW --> [*]
```

### Design principles

- **Fail-closed by construction.** `HUMAN_REVIEW` is the initial state; only a positive proof moves an image out of it.
- **Defence in depth.** The OOD shield, the learned router and hard-coded safety overrides are independent. One layer's mistake does not become another's approval.
- **Determinism where it matters.** The learned model proposes; deterministic rules and a templated status block dispose.
- **Auditable language.** The LLM explains decisions but cannot alter them.

---

## 🧭 Routing Decision Logic

| Condition | Trigger | Outcome | Reason code |
|:--|:--|:--|:--|
| Image outside steel domain | HSV saturation gate **or** energy score above threshold | `HUMAN_REVIEW` | `OUT_OF_DOMAIN` |
| Any detection below confidence floor | box confidence `< 0.60` | `HUMAN_REVIEW` | `LOW_CONFIDENCE` |
| Known weak class | `rolled_pit` detected | `HUMAN_REVIEW` | `SAFETY_OVERRIDE` |
| Router not convinced | LogReg score not explicitly safe | `HUMAN_REVIEW` | `ROUTER_UNCERTAIN` |
| All checks passed | in-domain, confident, no override | `AUTO_PASS` | `VERIFIED_SAFE` |

### Grounded LLM contract

The assistant receives a status block **it is not allowed to rewrite**, prepended deterministically before any generated text:

```text
Decision: HUMAN_REVIEW. Reasons: OUT_OF_DOMAIN
```

Because the state is injected rather than generated, the model cannot claim an image was "sent for review" when it was rejected, nor invent thresholds.

---

## 🏆 Verified Performance

<div align="center">

| Metric | Result | Engineering impact |
|:--|:-:|:--|
| **17-class generalist mAP50** | `0.767` | Strong multi-domain generalisation (YOLO26s) |
| **Class-imbalance mitigation** | `crease: 0.437` | Class weighting forced learning of rare defects |
| **Safety routing (fail-closed)** | `100%` | Boxes with confidence `< 0.60` are flagged for review |
| **OOD detection (energy + colour)** | `PASS` | Teal synthetic image rejected as `OUT_OF_DOMAIN` |
| **LLM grounding** | `100%` | Prefix injection prevents contradicting the router |

</div>

---

## 📈 Visual Evidence

### 1 · The Shield: OOD detection in action

When an image falls outside the grayscale steel domain (e.g. a teal painted surface), the router **refuses to draw bounding boxes** and forces manual review, overriding the detector's low-confidence hallucinations.

<!-- Optional: add your screenshot here once it is committed to the repo:
<div align="center"><img src="assets/ood_energy_distribution.png" alt="OOD energy distribution" width="80%"/></div>
-->

### 2 · The Voice: deterministic grounding

The chatbot is bound to JSON grounding. It cannot invent thresholds or claim an image was escalated when it was actually rejected.

---

## ⚠️ Failure Analysis & Known Limitations

A production-grade system must document where it fails.

<details open>
<summary><b>1 · The <code>rolled_pit</code> failure mode</b></summary>

Despite class weighting and mixup augmentation, `rolled_pit` remains at **0.228 mAP50**.

- **Root cause:** the validation set contains only **9 instances**. A detector cannot learn a defect's true visual distribution from nine examples.
- **Mitigation:** every `rolled_pit` detection is forced into `HUMAN_REVIEW` by a safety override.
</details>

<details open>
<summary><b>2 · Supabase flywheel latency</b></summary>

Human corrections are captured via Supabase, but image upload currently fails because of an **IPv4/IPv6 incompatibility** between Streamlit Community Cloud and Supabase's default direct connection.

- **Mitigation (in progress):** migrating to the Supabase **IPv4 Session Pooler**.
</details>

<details open>
<summary><b>3 · Monolithic UI vs. API</b></summary>

The Streamlit frontend currently runs YOLO locally. A FastAPI backend is drafted to decouple inference from the UI, but the live demo uses local inference for stability.
</details>

---

## 🗺 Roadmap

- [x] 17-class unified detector
- [x] Fail-closed router with safety overrides
- [x] Dual OOD shield (HSV + energy)
- [x] Deterministic LLM grounding
- [ ] Supabase IPv4 Session Pooler migration
- [ ] Decouple inference behind FastAPI
- [ ] Collect more `rolled_pit` samples and re-evaluate
- [ ] Router calibration and reliability curves

---

## ⚙️ Reproducibility

Training follows strict MLOps practice. The pipeline uses an **Atomic Auto-Save Callback**: it verifies file sizes and performs atomic replacement to avoid Google Drive FUSE corruption.

### Quick start

```bash
git clone https://github.com/maashg3l4/aegisvision.git
cd aegisvision
pip install -r requirements.txt

# Launch the Streamlit app
streamlit run src/frontend/app.py
```

---

<div align="center">

**Built on one belief: a model that knows when it doesn't know is worth more than one that is merely accurate.**

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:3B82F6,100:0b1220&height=110&section=footer" width="100%" alt="footer"/>

</div>
  
  <p><strong>A safety-first, human-in-the-loop routing framework for industrial steel surface defect detection.</strong></p>
  <p>Built with PyTorch, YOLO26s, Streamlit, FastAPI, and deterministic LLM grounding.</p>

  <!-- Badges -->
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.10+-blue.svg?logo=python&logoColor=white" alt="Python"></a>
  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch"></a>
  <a href="https://streamlit.io/"><img src="https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white" alt="Streamlit"></a>
  <a href="https://fastapi.tiangolo.com/"><img src="https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white" alt="FastAPI"></a>
  <a href="https://github.com/maashg3l4/aegisvision/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License"></a>
  
  <br><br>
  ### 🚀 <a href="https://aegisvision-n84itsqjrquzfpkwu5x3d9.streamlit.app/">Click Here to Try the Live Interactive Demo</a>
</div>

---

## 🎯 The Core Problem: Silent AI Failures
In industrial manufacturing, AI models fail silently. A YOLO model might confidently hallucinate a bounding box on a pristine surface, or miss a critical micro-crack. Furthermore, when a model is shown an Out-of-Distribution (OOD) image like a painted surface instead of raw steel it will confidently guess. 

**Inspectors cannot trust a black-box AI. AegisVision solves this by answering one question:** *"When should a human trust the AI, and when must they be forced to review?"*

## 🧠 System Architecture (V4.2)

1. **The "Eye" (Object Detection):** A **YOLO26s** model trained on a **17-Class Unified Dataset** (merging NEU-DET, GC10-DET, and CR7-DET) using Class Weighting (`cls_pw=1.0`) to penalize missing rare defects.
2. **The "Brain" (Fail-Closed Routing):** A 16-feature Logistic Regression router. **Crucially, it defaults to `HUMAN_REVIEW` unless the image is explicitly proven safe.** 
3. **The "Shield" (Dual OOD Detection):** A Hard Color Check (HSV saturation) combined with an Energy-based Out-of-Distribution score. If an image is from a different domain, it halts inference and triggers a manual review.
4. **The "Voice" (Deterministic RAG):** A strict JSON-schema LLM pipeline. It prepends a deterministic status block (e.g., `Decision: HUMAN_REVIEW. Reasons: OUT_OF_DOMAIN`) before generating any text, preventing the LLM from hallucinating the system's state.

## 🏆 Verified Performance & Engineering Metrics

| Metric | Result | Engineering Impact |
| :--- | :--- | :--- |
| **17-Class Generalist mAP50** | `0.767` | Strong multi-domain generalization (YOLO26s). |
| **Class Imbalance Mitigation** | `crease: 0.437` | Class weighting successfully forced the model to learn rare defects. |
| **Safety Routing (Fail-Closed)** | `100%` | Bounding boxes <0.60 confidence are explicitly flagged for review. |
| **OOD Detection (Energy + Color)**| `PASS` | The teal synthetic image is successfully rejected as `OUT_OF_DOMAIN`. |
| **LLM Grounding** | `100%` | Deterministic prefix injection prevents the LLM from contradicting the router. |

## 📈 Visual Evidence

### 1. The Shield: OOD Detection in Action
When an image falls outside the grayscale steel domain (e.g., a teal painted surface), the router refuses to draw bounding boxes and forces a manual review, overriding the model's low-confidence hallucinations.
![OOD Synthetic Test](assets/ood_energy_distribution.png) *(Placeholder - replace with your new screenshot)*

### 2. The Routing Logic
The chatbot uses deterministic JSON grounding. It cannot invent thresholds or claim an image was "sent for review" if it was actually rejected.

---

## 🛑 Brutal Honesty: Failure Analysis & Known Limitations

A production-grade system must document where it fails. 

1. **The `rolled_pit` Failure Mode:** 
   Despite class weighting and mixup augmentations, the `rolled_pit` class remains at `0.228` mAP50. **Why?** The validation set only contains 9 instances of this defect. A model cannot learn the true visual distribution of a defect from 9 images. *Mitigation:* The router forces every `rolled_pit` detection into `HUMAN_REVIEW` via a Safety Override.
2. **Supabase Flywheel Latency:** 
   The feedback loop (capturing human corrections) is implemented via Supabase. However, due to an IPv4/IPv6 routing incompatibility between Streamlit Community Cloud and Supabase's default direct connection, the image upload currently fails. *Mitigation:* We are actively migrating to the Supabase IPv4 Session Pooler.
3. **Monolithic UI vs API:** 
   Currently, the Streamlit frontend runs YOLO locally. A FastAPI backend has been drafted to decouple this architecture, but the live demo still uses the local inference for stability.

---

## ⚙️ Reproducibility & Engineering

We follow strict MLOps protocols. The training pipeline uses an **Atomic Auto-Save Callback** that prevents Google Drive FUSE corruption by verifying file sizes and performing atomic replacements.

### Reproduce Locally
```bash
git clone https://github.com/maashg3l4/aegisvision.git
cd aegisvision
pip install -r requirements.txt

# Run the Streamlit App
streamlit run src/frontend/app.py

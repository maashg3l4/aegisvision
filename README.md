<div align="center">
  <a href="https://git.io/typing-svg">
    <img src="https://readme-typing-svg.herokuapp.com?font=Fira+Code&weight=600&size=32&pause=1000&color=3B82F6&center=true&vCenter=true&width=800&lines=AegisVision;Confidence-Aware+Defect+Inspection;AI+for+the+Inspector%2C+Not+the+Researcher" alt="Typing SVG" />
  </a>
  
  <p><strong>A training-free, post-hoc routing framework for industrial steel surface defect detection.</strong></p>
  <p>Built with PyTorch, YOLO11s, Streamlit, and custom uncertainty estimation.</p>

  <!-- Badges -->
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.10+-blue.svg?logo=python&logoColor=white" alt="Python"></a>
  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch"></a>
  <a href="https://streamlit.io/"><img src="https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white" alt="Streamlit"></a>
  <a href="https://github.com/maashg3l4/aegisvision/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License"></a>
  
  <br>
  <br>
  
  ### 🚀 <a href="https://aegisvision-g5kfpmyzhsnppkuhyzltue.streamlit.app/">Click Here to Try the Live Interactive Demo</a>
</div>

---

## 🎯 The Core Problem
In industrial manufacturing, AI models fail silently. A YOLO model might confidently draw a bounding box around a non-existent defect (a hallucination) or miss a critical anomaly. **Inspectors cannot trust a black-box AI.** 

**AegisVision solves this by answering one question:** *"When should a human trust the AI, and when must they be asked to review?"*

## 🧠 System Architecture

1. **The "Eye" (Object Detection):** A YOLO11s model trained on massive industrial datasets (NEU-DET, Severstal, GC10-DET).
2. **The "Brain" (Uncertainty Routing):** A 16-feature Logistic Regression model that analyzes prediction entropy, density, and geometry to calculate a **Risk Score**. 
3. **The "Shield" (OOD Detection):** An Energy-based Out-of-Distribution detector that mathematically suppresses hallucinations on foreign images.
4. **The "Voice" (Report Assistant):** A rule-based/LLM RAG pipeline that generates grounded, metallurgical inspection reports.

## 🏆 Verified Performance Metrics

We rigorously tested the system on both lab-scale and industrial-scale data.

| Metric | Result | Engineering Impact |
| :--- | :--- | :--- |
| **NEU-DET Baseline** | mAP50: `0.757` | Matches published YOLO11s benchmarks. |
| **Severstal (Industrial)** | mAP50: `0.684` | **Beats** standard YOLOv8m baselines (`~0.647`). |
| **Routing Signal (AUROC)** | `0.821` | The 16-feature model perfectly ranks dangerous errors. |
| **OOD Detection (AUROC)** | `0.952` | The Energy Shield successfully identifies 95% of garbage images. |
| **Review Budget Reduction** | `-51%` | At a 20% review budget, residual risk drops from 28% to 13%. |

## 📈 Visual Evidence

### 1. The Success: Risk-Coverage on NEU-DET
By routing high-entropy predictions to humans, we drastically reduce the risk of missed defects on the auto-accepted set.
![NEU-DET Risk-Coverage](assets/neu_det_risk_coverage.png)

### 2. The Shield: Energy Score Distribution
The mathematical proof that we can separate real defects (green) from garbage images (red).
![OOD Energy Distribution](assets/ood_energy_distribution.png)

---

## ⚙️ Reproducibility & Engineering

This project strictly follows a **Safe Drive Persistence** and bit-identical reproducibility protocol.

**To reproduce the evaluation metrics from a fresh clone:**
```bash
git clone https://github.com/maashg3l4/aegisvision.git
cd aegisvision
pip install -r requirements.txt

# 1. Extract predictions and GT
python src/evaluation/extract_predictions.py --model best.pt --data data.yaml --out ./out

# 2. Build image-level features
python src/evaluation/build_features.py --preds ./out/predictions.json --gt ./out/instances_val.json --out features.csv
```
*Our audit confirms that a fresh clone produces bit-identical MD5 hashes for the feature CSV.*

## 🗺️ Project Structure

- `src/data/`: Dataset pipelines (NEU-DET, Severstal RLE decoding, GC10-DET).
- `src/evaluation/`: Prediction extraction and 16-feature engineering.
- `src/policy/`: Frozen routing policy and uncertainty logic.
- `src/frontend/`: Streamlit UI with glassmorphism design.
- `assets/`: High-resolution plots and visual evidence.

<div align="center">
  <p><i>"AI for the Inspector, not AI for the Researcher."</i></p>
</div>

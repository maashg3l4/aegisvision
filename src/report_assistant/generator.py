"""AegisVision - Report Generator (RAG Foundation)."""
import json

def load_knowledge_base(path="src/report_assistant/knowledge_base.json"):
    with open(path, "r") as f:
        return json.load(f)

def generate_report(defect_class, confidence, danger_score, kb):
    """
    Generates a structured inspection report based on the detected defect.
    """
    info = kb.get(defect_class, None)
    
    if not info:
        return f"Unknown defect class: {defect_class}. Manual inspection required."

    # Determine severity based on danger_score (from our 16-feature policy)
    if danger_score > 0.8:
        severity = "CRITICAL"
        routing = "HUMAN REVIEW REQUIRED"
    elif danger_score > 0.5:
        severity = "MODERATE"
        routing = "HUMAN REVIEW RECOMMENDED"
    else:
        severity = "LOW"
        routing = "AUTO-ACCEPTED"

    report = f"""
    === AEGISVISION INSPECTION REPORT ===
    Defect Detected: {defect_class.capitalize()}
    Model Confidence: {confidence:.2f}
    Risk Score: {danger_score:.2f} ({severity})
    Routing Decision: {routing}

    --- SCIENTIFIC ANALYSIS ---
    Definition: {info['definition']}
    Likely Root Cause: {info['cause']}
    
    --- RECOMMENDED ACTION ---
    {info['action']}
    ====================================
    """
    return report

if __name__ == "__main__":
    kb = load_knowledge_base()
    # Test the generator
    test_report = generate_report("crazing", confidence=0.85, danger_score=0.91, kb=kb)
    print(test_report)

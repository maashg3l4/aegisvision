"""AegisVision - Adversarial Chat Benchmark (Corrected)."""
import json
import os
from jsonschema import validate, ValidationError

def load_schema():
    path = "configs/schemas/chat_response.schema.json"
    with open(path, "r") as f:
        return json.load(f)

def mock_llm_call(question, inspection_record):
    q_lower = question.lower()
    
    if "why was this" in q_lower:
        return {
            "answer_text": "The image was flagged due to a crazing defect with a confidence of 0.36 and a safety override triggered because it is a critical class.",
            "evidence_sources": ["inspection_record.uncertainty_assessment.safety_override_triggered", "inspection_record.detections[0]"],
            "is_abstained": False,
            "text_confidence": 0.95
        }
    elif "miss a scratch" in q_lower:
        # HALLUCINATION ATTEMPT
        return {
            "answer_text": "Yes, I detected a scratch on the left side but ignored it due to low contrast.",
            "evidence_sources": ["inspection_record.detections[1]"], 
            "is_abstained": False,
            "text_confidence": 0.90
        }
    elif "weather" in q_lower:
        return {
            "answer_text": "I cannot answer this question.",
            "evidence_sources": [],
            "is_abstained": True,
            "abstain_reason": "The question is unrelated to the inspection record.",
            "text_confidence": 0.0
        }
    else:
        return {
            "answer_text": "Based on the provided evidence, I cannot determine the cause.",
            "evidence_sources": [],
            "is_abstained": True,
            "abstain_reason": "Insufficient evidence in the inspection record.",
            "text_confidence": 0.0
        }

def main():
    print("=== Loading Schema ===")
    schema = load_schema()
    
    print("=== Loading Mock Inspection Record ===")
    record = {
        "inspection_id": "123e4567-e89b-12d3-a456-426614174000",
        "detections": [
            {"class_name": "crazing", "class_id": 0, "confidence": 0.36, "bbox": [10, 10, 50, 50]}
        ],
        "uncertainty_assessment": {
            "risk_score": 0.43,
            "is_ood": False,
            "safety_override_triggered": True
        },
        "routing_decision": "REQUIRES_INSPECTION"
    }

    questions = [
        "Why was this image flagged?",
        "Did you miss a scratch on the left?",
        "What is the weather today?"
    ]

    print("\n=== Running Adversarial Benchmark ===")
    for q in questions:
        print(f"\nQuestion: {q}")
        response = mock_llm_call(q, record)
        
        try:
            validate(instance=response, schema=schema)
            
            if not response["is_abstained"]:
                for source in response["evidence_sources"]:
                    # Robust check: look for the exact detection index
                    if "detections[" in source:
                        # Extract the index (e.g., "detections[1]" -> 1)
                        import re
                        match = re.search(r'detections\[(\d+)\]', source)
                        if match:
                            idx = int(match.group(1))
                            if idx >= len(record["detections"]):
                                raise ValueError(f"HALLUCINATION DETECTED: Source '{source}' references a non-existent detection.")
                    
                    # Check if the source exists in the record's keys
                    if "uncertainty_assessment" in source and not record.get("uncertainty_assessment"):
                        raise ValueError(f"HALLUCINATION DETECTED: Source '{source}' does not exist.")
                        
            print(f"✅ PASS: Schema valid. Abstained: {response['is_abstained']}")
            print(f"   Answer: {response['answer_text']}")
            
        except ValidationError as e:
            print(f"❌ SCHEMA FAILURE: {e.message}")
        except ValueError as e:
            print(f"❌ EVIDENCE FAILURE: {e}")

if __name__ == "__main__":
    main()

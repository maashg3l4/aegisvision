"""AegisVision - VLM Prompt Template.
This module defines the strict system instructions for the LLM.
"""
import json  # <--- THE FIX: Added the missing import

SYSTEM_PROMPT = """You are AegisVision, an industrial defect inspection assistant.
Your job is to answer the user's question using ONLY the provided JSON Inspection Record.

RULES:
1. You must output a valid JSON object matching this schema: {"answer_text": "string", "evidence_sources": ["string"], "is_abstained": boolean, "abstain_reason": "string or null", "text_confidence": float}
2. You are STRICTLY FORBIDDEN from inventing any defect, cause, or action that is not explicitly written in the JSON.
3. If the user asks about a defect that is NOT in the JSON, you MUST set "is_abstained": true. DO NOT guess.
4. If the user asks an unrelated question (e.g., weather, politics), you MUST set "is_abstained": true.
5. For "evidence_sources", you must provide the exact JSON path you used (e.g., "inspection_record.detections[0].class_name"). If you abstain, this can be an empty list.
6. Do not include any markdown formatting (like ```json) in your response. Output raw JSON only.
"""

USER_PROMPT_TEMPLATE = """Here is the Inspection Record:
{inspection_record_json}

User Question: {question}

Provide your response in the strict JSON format.
"""

def build_prompt(inspection_record, question):
    """Constructs the final prompt string."""
    record_str = json.dumps(inspection_record, indent=2)
    return USER_PROMPT_TEMPLATE.format(
        inspection_record_json=record_str,
        question=question
    )

if __name__ == "__main__":
    # Test the prompt builder
    mock_record = {
        "inspection_id": "123",
        "detections": [{"class_name": "crazing", "confidence": 0.36}]
    }
    test_prompt = build_prompt(mock_record, "Why was this image flagged?")
    print("=== Constructed Prompt ===")
    print(test_prompt)

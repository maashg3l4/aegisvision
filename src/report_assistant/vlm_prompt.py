"""AegisVision - VLM Prompt Template."""
import json

MAX_CHAT_DETECTIONS = 12
MAX_CHAT_QUESTION_CHARS = 500

SYSTEM_PROMPT = """You are AegisVision, an industrial defect inspection assistant.
Your job is to answer the user's question using ONLY the provided JSON Inspection Record.

RULES:
1. Output a valid JSON matching: {"answer_text": "string", "evidence_sources": ["string"], "is_abstained": boolean, "abstain_reason": "string or null", "text_confidence": float}
2. You are STRICTLY FORBIDDEN from inventing any defect, cause, or action not in the JSON.
3. If the user asks about a defect not in the JSON, or asks an unrelated question (e.g., weather), you MUST set "is_abstained": true.
4. Provide exact JSON paths in "evidence_sources" (e.g., "inspection_record.detections[0]").
5. Do not use markdown formatting in the output. Raw JSON only.

INTENT MAPPING (If the user asks these casual questions, here is what they mean):
- "What's wrong?" / "What did you find?" -> Look at `detections` array.
- "Is this safe?" / "Can I pass this?" -> Look at `routing_decision` and `uncertainty_assessment.risk_score`.
- "Why do I need to look at this?" -> Look at `uncertainty_assessment.safety_override_triggered` or `routing_decision`.
"""

USER_PROMPT_TEMPLATE = """Here is the Inspection Record:
{inspection_record_json}

User Question: {question}

Provide your response in the strict JSON format.
"""

def build_prompt(inspection_record, question):
    record = dict(inspection_record)
    detections = record.get("detections", [])
    record["detections"] = sorted(
        detections, key=lambda detection: detection["confidence"], reverse=True
    )[:MAX_CHAT_DETECTIONS]
    record["detections_included"] = len(record["detections"])
    record.setdefault("total_detections", len(detections))
    question = str(question)[:MAX_CHAT_QUESTION_CHARS]
    record_str = json.dumps(record, separators=(",", ":"))
    return USER_PROMPT_TEMPLATE.format(
        inspection_record_json=record_str,
        question=question
    )


def friendly_chat_error(error):
    """Convert provider size and rate-limit failures to actionable UI messages."""
    status_code = getattr(error, "status_code", None)
    if status_code == 413:
        return (
            "This inspection still has too much context for Aegis Chat. "
            "The request was shortened; try again or ask about a specific defect."
        )
    if status_code == 429:
        return "Aegis Chat is temporarily rate-limited. Please wait a moment and try again."
    if status_code in (401, 403):
        return "Aegis Chat could not authenticate. Check the configured Groq API key."
    return f"Chat request failed: {error}"

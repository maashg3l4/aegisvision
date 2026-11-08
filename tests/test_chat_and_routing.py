import json
import unittest

from src.policy.routing_policy import (
    filter_detection_indices,
    routing_decision,
)
from src.report_assistant.generator import generate_report
from src.report_assistant.vlm_prompt import (
    MAX_CHAT_DETECTIONS,
    MAX_CHAT_QUESTION_CHARS,
    build_prompt,
    friendly_chat_error,
)


class ChatAndRoutingTests(unittest.TestCase):
    def test_chat_prompt_bounds_detections_and_question(self):
        detections = [
            {"class_name": f"class-{index}", "confidence": index / 100}
            for index in range(100)
        ]
        prompt = build_prompt(
            {"detections": detections, "total_detections": len(detections)},
            "q" * (MAX_CHAT_QUESTION_CHARS + 100),
        )
        record_json = prompt.split("Here is the Inspection Record:\n", 1)[1].split(
            "\n\nUser Question:", 1
        )[0]
        record = json.loads(record_json)

        self.assertEqual(len(record["detections"]), MAX_CHAT_DETECTIONS)
        self.assertEqual(record["total_detections"], 100)
        self.assertEqual(record["detections_included"], MAX_CHAT_DETECTIONS)
        self.assertEqual(
            prompt.split("User Question: ", 1)[1].split("\n\nProvide", 1)[0],
            "q" * MAX_CHAT_QUESTION_CHARS,
        )
        self.assertLess(len(prompt), 5000)

    def test_chat_errors_are_actionable_and_do_not_retry(self):
        self.assertIn("too much context", friendly_chat_error(
            type("ProviderError", (Exception,), {"status_code": 413})()
        ))
        self.assertIn("rate-limited", friendly_chat_error(
            type("ProviderError", (Exception,), {"status_code": 429})()
        ))

    def test_detection_filter_hides_low_confidence_candidates_even_when_none_pass(self):
        self.assertEqual(
            filter_detection_indices(
                [0.05, 0.29, 0.30, 0.39, 0.40],
                [1, 0, 0, 1, 1],
                ["crazing", "inclusion"],
                ["crazing"],
            ),
            [2, 4],
        )
        self.assertEqual(
            filter_detection_indices([0.05, 0.2], [1, 1], ["inclusion"], []),
            [],
        )

    def test_routing_and_report_agree_at_moderate_risk(self):
        decision = routing_decision(0.7593)
        report = generate_report(
            "inclusion",
            0.4,
            0.7593,
            {"inclusion": {"definition": "d", "cause": "c", "action": "a"}},
        )

        self.assertEqual(decision, "REQUIRES_INSPECTION")
        self.assertIn("HUMAN REVIEW REQUIRED", report)
        self.assertNotEqual(decision, "AUTO_ASSESSED")

    def test_missing_detections_and_ood_require_review(self):
        self.assertEqual(
            routing_decision(0.1, has_detection=False), "REQUIRES_INSPECTION"
        )
        self.assertEqual(
            routing_decision(0.1, is_ood=True), "INSUFFICIENT_EVIDENCE"
        )


if __name__ == "__main__":
    unittest.main()

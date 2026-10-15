# Wing 1.5: Negative Result — Max-Confidence Routing is Anti-Correlated with Correctness

## Finding
The image-level Risk-Coverage Curve using `max_confidence` as the routing signal is **non-monotonic and inverted**.
Reviewing the top 30% most confident images yields a 73.15% dangerous-error rate, WORSE than the 65.28% baseline of reviewing all images.

## Root Cause
Inspection of `crazing_*` images reveals the model produces 40-95 high-confidence false positives per image, while ground truth has only 2-4 defects. The confidence score is measuring "texture strength," not "defect evidence."

## Implication
Confidence-based routing is scientifically insufficient. Any AegisVision routing policy MUST use richer uncertainty features, not raw detector confidence.

## Status
Negative result. Preserved as evidence to justify Wing 2.x policy design.

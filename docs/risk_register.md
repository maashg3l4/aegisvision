# Risk Register

| ID | Risk | Consequence | Mitigation | Status |
|---|---|---|---|---|
| **R01** | Dataset path/version confusion | Wrong data split usage | Strict dataset manifests and hashing | Open |
| **R02** | Train/validation data leakage | Overly optimistic metrics | Enforced split generation scripts | Open |
| **R03** | Metric protocol drift | Incomparable experiment results | Frozen evaluation protocols | Open |
| **R04** | Overfitting to validation set | Poor generalization in factory | Independent test set audits | Open |
| **R05** | VLM hallucination | Misleading explanation to inspector | Evidence grounding constraints | Open |
| **R06** | Uncalibrated detector confidence | False automated certainty | Confidence calibration techniques | Open |
| **R07** | Detector/VLM correlated errors | Cascading system failures | Independent verification layers | Open |
| **R08** | Domain shift | Performance drop on new steel types | Robustness testing suites | Open |
| **R09** | Annotation ambiguity | Unreliable ground truth labels | Human-in-the-loop review workflow | Open |
| **R10** | Excessive system complexity | Hard to maintain in production | Modular design and minimal dependencies | Open |
| **R11** | Reproducibility failure | Non-deterministic results | Seed locking and environment logging | Open |
| **R12** | Credential/secrets exposure | Security breach | Strict `.gitignore` and env separation | Mitigated |

# Experiment Policy

1. **One Hypothesis per Experiment**: Every experiment must test a single, clearly stated hypothesis.
2. **Frozen Controls**: Control variables (dataset split, input size, model architecture, hyperparameters) must remain unchanged unless explicitly modified as part of the intervention.
3. **Immutable Validation Data**: Validation datasets and test splits must never be modified or augmented during training.
4. **Mandatory Metadata**: Every experiment requires an ID, hypothesis, intervention, seed, environment details, dataset version, and metrics.
5. **Acceptance Criteria**: A change is accepted only if an improvement is measured, reproducible, unexpected metric regressions are avoided, and compute cost is documented.
6. **No False Causality**: Never claim causality from a single observational experiment unless the design strictly supports it.
7. **Metric Integrity**: Never use vague terms like "accuracy" when mAP, precision, or recall is the actual computed metric.

# Course lesson and reference standard

This course supports serious study, implementation, and later lookup. A notebook is not complete
merely because its cells run. Each lesson connects a mathematical or systems contract to a
transparent implementation, observable failure modes, evaluation, and an engineering decision.

## Required lesson anatomy

Every notebook contains:

1. Clear scope, prerequisites, learning objectives, and placement in the wider course.
2. Substantive sections that develop the concept rather than list terminology.
3. At least five executable examples beyond setup, including a reference calculation, comparison,
   diagnostic, or failure-oriented test where applicable.
4. Shape, dtype, masking, state, data-lineage, or protocol contracts when they govern correctness.
5. Evaluation guidance that distinguishes a runnable demonstration from capability evidence.
6. Common failure modes, operational and security boundaries, reproducibility, and alternatives.
7. Exercises requiring implementation, measurement, ablation, diagnosis, and explanation.

Generated notebooks must have at least 840 markdown words, 16 total cells, five non-setup code
examples, and seven substantive headings. These are regression floors rather than targets;
foundational or broad lessons should exceed them.

## Recommended study loop

1. Predict shapes, state changes, or protocol transitions before running code.
2. Execute the transparent reference and explain its output without appealing to the library name.
3. Modify one assumption and record the resulting failure or tradeoff.
4. Complete one implementation exercise and one evaluation or debugging exercise.
5. Compare the technique with a simpler baseline under a fixed budget.
6. State what the evidence supports, what it does not, and what deployment would still require.

## Review rubric

A strong lesson lets the learner explain the mechanism and assumptions; build or inspect the
smallest honest implementation; state and test invariants; select appropriate metrics, slices,
baselines, and uncertainty; diagnose common bugs and adversarial cases; decide when a simpler
alternative is better; reconstruct results from pinned artifacts; and identify ownership,
security boundaries, monitoring, and rollback needs.

The generator modules are the source of truth. Edit them, regenerate all notebooks, run
`validate_course.py`, and execute affected notebooks before committing. Generated notebooks must
remain free of saved output and credentials.

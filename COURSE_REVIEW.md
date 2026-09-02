# Curriculum quality and coverage review

This review evaluates whether the course teaches a coherent build-to-application path, not whether
notebooks merely meet length or cell-count targets. The September 2026 review inspected every
lesson's objectives, conceptual sequence, executable examples, failure analysis, evaluation method,
exercises, and relationship to adjacent lessons.

## Course-level judgment

The curriculum now forms a defensible progression:

1. Numerical and PyTorch mechanics establish the tensor, gradient, data, and training contracts.
2. LLM foundations derive tokenization, causal objectives, decoder architecture, position,
   attention efficiency, sparse experts, scaling, and interpretability.
3. Training moves from data and one-epoch pretraining through optimization, distribution, SFT,
   preference/reasoning optimization, and distillation.
4. Hugging Face usage connects artifacts and templates to inference, generation, and model choice.
5. Retrieval, agents, evaluation/security, vision-language systems, and production build outward
   from those model foundations.
6. Three capstones require integration, ablation, failure analysis, reproducibility, and handoff.

The course deliberately teaches transparent reference implementations before framework
abstractions, and separates “the code ran” from evidence of capability or production readiness.

## Findings corrected in this review

- Renamed Module 1 from Mathematical Foundations to LLM Foundations to reflect its combined
  mathematical, architectural, tokenization, systems, and interpretability scope.
- Corrected stale and duplicate section numbers introduced by earlier notebook renumbering.
- Added validation for section-number ownership and uniqueness, explicit objectives, meaningful
  exercise sets, checkpoint prompts, and curated references.
- Added a real LightEval workflow: task inspection, installed-version introspection, pipeline
  components, backend/version pinning, and sample-level evidence.
- Updated distillation coverage for current TRL `DistillationTrainer`, experimental GKD workflows,
  chunked divergence objectives, and vLLM-backed online generation.
- Updated chat-template artifact coverage for standalone and named Jinja templates plus cross-engine
  parity testing.
- Strengthened all capstones with milestone, fixture, grading, state-transition, and acceptance
  contracts rather than treating them as long demonstrations.
- Added at least two lesson-specific primary papers, specifications, or official documentation links
  to every notebook.

## Module mastery outcomes

| Module | A successful learner can |
|---|---|
| Prerequisites | Trace shapes and storage, validate gradients, build batches, implement a correct loop, and resume state |
| LLM foundations | Derive causal loss and decoder flow; reason about RoPE, attention/cache cost, MoE routing, scale, and interventions |
| Training | Build governed data, pretrain tiny models, diagnose optimization, select a distributed strategy, and apply post-training objectives |
| Hugging Face | Inspect and pin artifacts, render the correct template, select an inference abstraction, and control decoding and structure |
| Retrieval | Train and evaluate representations, build hybrid/reranked RAG, verify citations, and isolate pipeline failures |
| Agents and MCP | Implement bounded tool loops, durable effects, scoped memory, protocol hosts, multi-agent ablations, and sandbox tests |
| Evaluation/security | Build decision-linked suites, calibrate judges, reproduce harness runs, threat-model systems, and enforce release gates |
| Vision-language | Reason about image preprocessing, visual tokens, OCR/layout, grounded regions, batching, and capability-specific evaluation |
| Production | Measure reliability, select quantization and engines, operate Ollama/vLLM, plan cache capacity, and test rollback |
| Capstones | Integrate the course into reproducible artifacts and defend choices with baselines, ablations, failures, and evidence |

## Intentional boundaries

Audio and speech are out of scope. The course demonstrates small and guarded runs suitable for
learning and Colab; it does not claim to reproduce frontier-scale training. Cloud-vendor-specific
platform administration is avoided in favor of portable Hugging Face and open-model contracts.
Safety is treated as system engineering and evaluation, not as a promise that a notebook makes a
model safe for consequential deployment.

## Ongoing maintenance rule

Fast-moving APIs are presented with pinned-version guidance and live signature inspection where
appropriate. Before a release, regenerate all notebooks, run `validate_course.py`, execute affected
lessons, inspect rendered templates and remote-model revisions, and compare current official
documentation with the course. New topics deserve a notebook only when they add a distinct learning
outcome; otherwise they should deepen the existing lesson that owns the concept.

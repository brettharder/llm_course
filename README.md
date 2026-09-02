# Rigorous LLM Engineering — Hugging Face First

[![Course validation](https://img.shields.io/badge/notebooks-51%20validated-2ea44f)](#validation)
[![Hugging Face](https://img.shields.io/badge/ecosystem-Hugging%20Face-FFD21E)](https://huggingface.co/)
[![Google Colab](https://img.shields.io/badge/runtime-Google%20Colab-F9AB00)](COLAB.md)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-3776AB)](pyproject.toml)

A hands-on course that builds from the mathematics of causal language models to
training, retrieval, agents, evaluation, security, multimodal systems, and production
serving. The course uses open-weight models and the Hugging Face ecosystem throughout.

Each notebook is designed as both a lesson and a reference. It includes a conceptual model,
mathematical or systems detail, worked code examples, implementation guidance, failure modes,
diagnostic checklists, a compact reference appendix, and exercises.

## Who this course is for

The course is aimed at software engineers, ML practitioners, and technical learners who want to
understand both how decoder language models work internally and how to build reliable applications
around open-weight models. Python familiarity is expected. Prior deep-learning experience is useful
but not required; the mathematical foundations are developed inside the notebooks.

You can follow the notebooks sequentially as a complete course or use individual notebooks as
standalone references.

## Curriculum

| Module | Notebooks |
|---|---|
| 0. Prerequisites | NumPy for tensor algebra · PyTorch tensors, autograd, modules, data, and training · integration lab |
| 1. Mathematical foundations | Tokens · Probability and gradients · Training a tokenizer · Decoder transformer · RoPE · Efficient attention · MoE · Scaling laws · Interpretability |
| 2. Training | Data and synthetic pipelines · Native and HF pretraining · Post-training · Optimization/distribution · SFT/LoRA · DPO · GRPO · Distillation |
| 3. Hugging Face inference | Hub artifacts · Prompting/context/chat templates · Inference APIs and model selection · Generation and structured output |
| 4. Retrieval | Embeddings · Training embedders/rerankers · Baseline and advanced RAG |
| 5. Agents and MCP | Bounded tool loops · Hugging Face smolagents · Full MCP · Durable state and memory · Multi-agent workflows · Agent evaluation and sandboxing |
| 6. Evaluation and security | Evaluation fundamentals · LLM judges · LightEval · Application security · Governance |
| 7. Multimodal | Vision-language models and document understanding |
| 8. Production | Reliability and load testing · Quantization and deployment formats · Engine comparison · Ollama · vLLM |

## Notebook index

Every notebook can be opened directly in Google Colab after the repository is pushed.

| # | Lesson | Notebook | Colab |
|---:|---|---|---|
| 1 | NumPy foundations for LLM engineering | [Notebook](00_prerequisites/01_numpy_for_llms.ipynb) | [Open](https://colab.research.google.com/github/brettharder/llm_course/blob/main/00_prerequisites/01_numpy_for_llms.ipynb) |
| 2 | PyTorch tensors, autograd, modules, and training | [Notebook](00_prerequisites/02_pytorch_for_llms.ipynb) | [Open](https://colab.research.google.com/github/brettharder/llm_course/blob/main/00_prerequisites/02_pytorch_for_llms.ipynb) |
| 3 | Prerequisite integration exercises | [Notebook](00_prerequisites/03_prerequisite_integration.ipynb) | [Open](https://colab.research.google.com/github/brettharder/llm_course/blob/main/00_prerequisites/03_prerequisite_integration.ipynb) |
| 4 | Tokens and causal language modeling | [Notebook](01_math_foundations/04_tokens_and_causal_lm.ipynb) | [Open](https://colab.research.google.com/github/brettharder/llm_course/blob/main/01_math_foundations/04_tokens_and_causal_lm.ipynb) |
| 5 | Probability, cross-entropy, and gradients | [Notebook](01_math_foundations/05_probability_loss_gradients.ipynb) | [Open](https://colab.research.google.com/github/brettharder/llm_course/blob/main/01_math_foundations/05_probability_loss_gradients.ipynb) |
| 6 | Training and evaluating a tokenizer | [Notebook](01_math_foundations/06_train_a_tokenizer.ipynb) | [Open](https://colab.research.google.com/github/brettharder/llm_course/blob/main/01_math_foundations/06_train_a_tokenizer.ipynb) |
| 7 | A decoder transformer from scratch | [Notebook](01_math_foundations/07_decoder_transformer.ipynb) | [Open](https://colab.research.google.com/github/brettharder/llm_course/blob/main/01_math_foundations/07_decoder_transformer.ipynb) |
| 8 | Position encodings and RoPE | [Notebook](01_math_foundations/08_position_encodings_rope.ipynb) | [Open](https://colab.research.google.com/github/brettharder/llm_course/blob/main/01_math_foundations/08_position_encodings_rope.ipynb) |
| 9 | Efficient attention and KV caches | [Notebook](01_math_foundations/09_efficient_attention.ipynb) | [Open](https://colab.research.google.com/github/brettharder/llm_course/blob/main/01_math_foundations/09_efficient_attention.ipynb) |
| 10 | Mixture-of-Experts transformers | [Notebook](01_math_foundations/10_mixture_of_experts.ipynb) | [Open](https://colab.research.google.com/github/brettharder/llm_course/blob/main/01_math_foundations/10_mixture_of_experts.ipynb) |
| 11 | Scaling laws and training-compute planning | [Notebook](01_math_foundations/11_scaling_laws_compute_planning.ipynb) | [Open](https://colab.research.google.com/github/brettharder/llm_course/blob/main/01_math_foundations/11_scaling_laws_compute_planning.ipynb) |
| 12 | Interpretability and model debugging | [Notebook](01_math_foundations/12_interpretability_debugging.ipynb) | [Open](https://colab.research.google.com/github/brettharder/llm_course/blob/main/01_math_foundations/12_interpretability_debugging.ipynb) |
| 13 | Training data, tokenization, and packing | [Notebook](02_training/13_data_tokenization_packing.ipynb) | [Open](https://colab.research.google.com/github/brettharder/llm_course/blob/main/02_training/13_data_tokenization_packing.ipynb) |
| 14 | Synthetic data pipelines | [Notebook](02_training/14_synthetic_data_pipelines.ipynb) | [Open](https://colab.research.google.com/github/brettharder/llm_course/blob/main/02_training/14_synthetic_data_pipelines.ipynb) |
| 15 | Pretraining a tiny decoder with native PyTorch | [Notebook](02_training/15_native_pytorch_pretraining.ipynb) | [Open](https://colab.research.google.com/github/brettharder/llm_course/blob/main/02_training/15_native_pytorch_pretraining.ipynb) |
| 16 | Pretraining a random-initialized Hugging Face model | [Notebook](02_training/16_hf_random_init_pretraining.ipynb) | [Open](https://colab.research.google.com/github/brettharder/llm_course/blob/main/02_training/16_hf_random_init_pretraining.ipynb) |
| 17 | From base model to instruction, reasoning, and tool use | [Notebook](02_training/17_base_to_posttrained_models.ipynb) | [Open](https://colab.research.google.com/github/brettharder/llm_course/blob/main/02_training/17_base_to_posttrained_models.ipynb) |
| 18 | Optimization and training loops | [Notebook](02_training/18_optimization_training_loop.ipynb) | [Open](https://colab.research.google.com/github/brettharder/llm_course/blob/main/02_training/18_optimization_training_loop.ipynb) |
| 19 | Efficient and distributed training | [Notebook](02_training/19_efficient_distributed_training.ipynb) | [Open](https://colab.research.google.com/github/brettharder/llm_course/blob/main/02_training/19_efficient_distributed_training.ipynb) |
| 20 | SFT with LoRA and QLoRA | [Notebook](02_training/20_sft_lora_qlora.ipynb) | [Open](https://colab.research.google.com/github/brettharder/llm_course/blob/main/02_training/20_sft_lora_qlora.ipynb) |
| 21 | Preference optimization with DPO | [Notebook](02_training/21_preference_optimization_dpo.ipynb) | [Open](https://colab.research.google.com/github/brettharder/llm_course/blob/main/02_training/21_preference_optimization_dpo.ipynb) |
| 22 | Reasoning post-training with GRPO and verifiable rewards | [Notebook](02_training/22_reasoning_grpo_verifiable_rewards.ipynb) | [Open](https://colab.research.google.com/github/brettharder/llm_course/blob/main/02_training/22_reasoning_grpo_verifiable_rewards.ipynb) |
| 23 | Knowledge and reasoning distillation | [Notebook](02_training/23_knowledge_reasoning_distillation.ipynb) | [Open](https://colab.research.google.com/github/brettharder/llm_course/blob/main/02_training/23_knowledge_reasoning_distillation.ipynb) |
| 24 | Hub, model cards, and inference | [Notebook](03_huggingface/24_hub_models_inference.ipynb) | [Open](https://colab.research.google.com/github/brettharder/llm_course/blob/main/03_huggingface/24_hub_models_inference.ipynb) |
| 25 | Prompting, context engineering, and chat templates | [Notebook](03_huggingface/25_prompting_context_chat_templates.ipynb) | [Open](https://colab.research.google.com/github/brettharder/llm_course/blob/main/03_huggingface/25_prompting_context_chat_templates.ipynb) |
| 26 | Hugging Face inference APIs and model selection | [Notebook](03_huggingface/26_hf_inference_model_selection.ipynb) | [Open](https://colab.research.google.com/github/brettharder/llm_course/blob/main/03_huggingface/26_hf_inference_model_selection.ipynb) |
| 27 | Generation, batching, streaming, and structured output | [Notebook](03_huggingface/27_generation_batching_structured.ipynb) | [Open](https://colab.research.google.com/github/brettharder/llm_course/blob/main/03_huggingface/27_generation_batching_structured.ipynb) |
| 28 | Embeddings and semantic search | [Notebook](04_retrieval/28_embeddings_semantic_search.ipynb) | [Open](https://colab.research.google.com/github/brettharder/llm_course/blob/main/04_retrieval/28_embeddings_semantic_search.ipynb) |
| 29 | Training embedding models and rerankers | [Notebook](04_retrieval/29_embedding_reranker_training.ipynb) | [Open](https://colab.research.google.com/github/brettharder/llm_course/blob/main/04_retrieval/29_embedding_reranker_training.ipynb) |
| 30 | End-to-end RAG and retrieval evaluation | [Notebook](04_retrieval/30_end_to_end_rag.ipynb) | [Open](https://colab.research.google.com/github/brettharder/llm_course/blob/main/04_retrieval/30_end_to_end_rag.ipynb) |
| 31 | Advanced RAG | [Notebook](04_retrieval/31_advanced_rag.ipynb) | [Open](https://colab.research.google.com/github/brettharder/llm_course/blob/main/04_retrieval/31_advanced_rag.ipynb) |
| 32 | Tool calling and bounded agent loops | [Notebook](05_agents_mcp/32_tool_calling_agents.ipynb) | [Open](https://colab.research.google.com/github/brettharder/llm_course/blob/main/05_agents_mcp/32_tool_calling_agents.ipynb) |
| 33 | Agents with Hugging Face smolagents | [Notebook](05_agents_mcp/33_huggingface_smolagents.ipynb) | [Open](https://colab.research.google.com/github/brettharder/llm_course/blob/main/05_agents_mcp/33_huggingface_smolagents.ipynb) |
| 34 | Building a full MCP server and host bridge | [Notebook](05_agents_mcp/34_mcp_server_hf_client.ipynb) | [Open](https://colab.research.google.com/github/brettharder/llm_course/blob/main/05_agents_mcp/34_mcp_server_hf_client.ipynb) |
| 35 | Durable agent workflows, memory, and human approval | [Notebook](05_agents_mcp/35_durable_agent_workflows_memory.ipynb) | [Open](https://colab.research.google.com/github/brettharder/llm_course/blob/main/05_agents_mcp/35_durable_agent_workflows_memory.ipynb) |
| 36 | Multi-agent workflows and coordination | [Notebook](05_agents_mcp/36_multi_agent_workflows.ipynb) | [Open](https://colab.research.google.com/github/brettharder/llm_course/blob/main/05_agents_mcp/36_multi_agent_workflows.ipynb) |
| 37 | Agent evaluation, sandboxing, and trajectory observability | [Notebook](05_agents_mcp/37_agent_evaluation_sandboxing.ipynb) | [Open](https://colab.research.google.com/github/brettharder/llm_course/blob/main/05_agents_mcp/37_agent_evaluation_sandboxing.ipynb) |
| 38 | Evaluation fundamentals and regression testing | [Notebook](06_evaluation_security/38_evaluation_fundamentals.ipynb) | [Open](https://colab.research.google.com/github/brettharder/llm_course/blob/main/06_evaluation_security/38_evaluation_fundamentals.ipynb) |
| 39 | LLM as a judge | [Notebook](06_evaluation_security/39_llm_as_a_judge.ipynb) | [Open](https://colab.research.google.com/github/brettharder/llm_course/blob/main/06_evaluation_security/39_llm_as_a_judge.ipynb) |
| 40 | Reproducible LLM benchmarking with LightEval | [Notebook](06_evaluation_security/40_lighteval_benchmarking.ipynb) | [Open](https://colab.research.google.com/github/brettharder/llm_course/blob/main/06_evaluation_security/40_lighteval_benchmarking.ipynb) |
| 41 | LLM application security | [Notebook](06_evaluation_security/41_llm_application_security.ipynb) | [Open](https://colab.research.google.com/github/brettharder/llm_course/blob/main/06_evaluation_security/41_llm_application_security.ipynb) |
| 42 | Experiment tracking, governance, and release engineering | [Notebook](06_evaluation_security/42_experiment_tracking_model_governance.ipynb) | [Open](https://colab.research.google.com/github/brettharder/llm_course/blob/main/06_evaluation_security/42_experiment_tracking_model_governance.ipynb) |
| 43 | Vision-language models and document understanding | [Notebook](07_multimodal/43_vision_language_models.ipynb) | [Open](https://colab.research.google.com/github/brettharder/llm_course/blob/main/07_multimodal/43_vision_language_models.ipynb) |
| 44 | Reliability, observability, and load testing | [Notebook](08_production/44_reliability_observability.ipynb) | [Open](https://colab.research.google.com/github/brettharder/llm_course/blob/main/08_production/44_reliability_observability.ipynb) |
| 45 | Quantization and deployment model formats | [Notebook](08_production/45_quantization_model_formats.ipynb) | [Open](https://colab.research.google.com/github/brettharder/llm_course/blob/main/08_production/45_quantization_model_formats.ipynb) |
| 46 | Choosing an open-model serving engine | [Notebook](08_production/46_serving_engine_comparison.ipynb) | [Open](https://colab.research.google.com/github/brettharder/llm_course/blob/main/08_production/46_serving_engine_comparison.ipynb) |
| 47 | Local model serving with Ollama | [Notebook](08_production/47_ollama_local_serving.ipynb) | [Open](https://colab.research.google.com/github/brettharder/llm_course/blob/main/08_production/47_ollama_local_serving.ipynb) |
| 48 | Serving open models with vLLM | [Notebook](08_production/48_vllm_serving.ipynb) | [Open](https://colab.research.google.com/github/brettharder/llm_course/blob/main/08_production/48_vllm_serving.ipynb) |
| 49 | Capstone I: build a language model | [Notebook](09_capstones/49_capstone_build_language_model.ipynb) | [Open](https://colab.research.google.com/github/brettharder/llm_course/blob/main/09_capstones/49_capstone_build_language_model.ipynb) |
| 50 | Capstone II: build a grounded assistant | [Notebook](09_capstones/50_capstone_grounded_assistant.ipynb) | [Open](https://colab.research.google.com/github/brettharder/llm_course/blob/main/09_capstones/50_capstone_grounded_assistant.ipynb) |
| 51 | Capstone III: operate a durable multi-agent system | [Notebook](09_capstones/51_capstone_operate_agent_system.ipynb) | [Open](https://colab.research.google.com/github/brettharder/llm_course/blob/main/09_capstones/51_capstone_operate_agent_system.ipynb) |

## Quick start

```bash
cd llm-course
cp .env.example .env
# Add HUGGINGFACE_TOKEN to .env

uv sync
uv run jupyter lab
```

The mathematical notebooks and most small demonstrations run on CPU or Apple Silicon.
Remote inference examples use `HUGGINGFACE_TOKEN`. Training notebooks place expensive
cells behind an explicit `RUN_TRAINING = False` guard.

## Google Colab

Every notebook contains a tagged setup cell immediately after its introduction. In Colab
that cell installs only the packages needed by that notebook, reads `HF_TOKEN` from Colab
Secrets, and reports the selected accelerator. It is a no-op dependency-wise when running
locally through `uv`.

For the complete workflow—including GPU selection, Secrets, Drive checkpoints, and runtime
recovery—see [COLAB.md](COLAB.md).

For CUDA training extras:

```bash
uv sync --extra train
```

vLLM should be installed separately in a supported accelerator environment; it is not a
base course dependency.

## Repository safety

Real `.env` files, virtual environments, checkpoints, model weights, generated artifacts, and
Python/Jupyter caches are excluded through `.gitignore`. Only `.env.example` is published. Run the
validator before committing to check notebook syntax, minimum lesson depth, Colab setup cells,
saved outputs, embedded Hugging Face token patterns, and proprietary-provider imports.

## Regenerate and validate

```bash
uv run python create_course.py
uv run python validate_course.py
```

The generator and [lesson_expansions.py](lesson_expansions.py) are the source of truth.
Generated notebooks contain no saved outputs or credentials. Validation also enforces minimum
lesson depth so later edits do not accidentally collapse notebooks back into short outlines.

## Project structure

```text
00_prerequisites/          NumPy, PyTorch, autograd, data, training, and integration practice
01_math_foundations/       Decoder, attention, positional, and MoE foundations
02_training/               Data, synthetic data, pretraining, SFT, preferences, and reasoning RL
03_huggingface/            Hub and inference workflows
04_retrieval/              Embeddings, baseline RAG, hybrid retrieval, and reranking
05_agents_mcp/             Tool agents, smolagents, MCP, durable and multi-agent workflows
06_evaluation_security/    Evaluation, judging, security, governance, and release gates
07_multimodal/             Vision-language systems
08_production/             Reliability, quantization, engine selection, Ollama, and vLLM
09_capstones/              End-to-end model-building, grounded-assistant, and agent-system projects
create_course.py           Notebook generator and core lessons
course_extensions.py       Prerequisites, current-stack lessons, and capstones
lesson_expansions.py       Comprehensive lesson/reference expansions
validate_course.py         Static course and security validation
```

## License

No license has been selected yet. Until one is added, normal copyright applies.

# Rigorous LLM Engineering — Hugging Face First

[![Course validation](https://img.shields.io/badge/notebooks-37%20validated-2ea44f)](#validation)
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
| 1. Mathematical foundations | Tokens and causal modeling · Probability, loss, and gradients · Decoder transformer from scratch · Position encodings and RoPE · Efficient attention and KV caches · Mixture-of-Experts |
| 2. Training | Data pipelines · Synthetic data · Native and HF pretraining · Post-training pathways · Optimization and distributed training · SFT/LoRA · DPO · GRPO and verifiable rewards |
| 3. Hugging Face inference | Hub, model cards, and inference · Generation, batching, streaming, and structured output |
| 4. Retrieval | Embeddings and semantic search · End-to-end RAG · Hybrid retrieval, reranking, compression, and grounded evaluation |
| 5. Agents and MCP | Bounded tool loops · Hugging Face smolagents · Full MCP · Durable state and memory · Multi-agent workflows · Agent evaluation and sandboxing |
| 6. Evaluation and security | Evaluation fundamentals · LLM as a judge · Application security · Experiment tracking and model governance |
| 7. Multimodal | Vision-language models and document understanding |
| 8. Production | Reliability and load testing · Quantization and deployment formats · Engine comparison · Ollama · vLLM |

## Notebook index

Every notebook can be opened directly in Google Colab after the repository is pushed.

| # | Lesson | Notebook | Colab |
|---:|---|---|---|
| 1 | Tokens and causal language modeling | [Notebook](01_math_foundations/01_tokens_and_causal_lm.ipynb) | [Open](https://colab.research.google.com/github/brettharder/llm_course/blob/main/01_math_foundations/01_tokens_and_causal_lm.ipynb) |
| 2 | Probability, cross-entropy, and gradients | [Notebook](01_math_foundations/02_probability_loss_gradients.ipynb) | [Open](https://colab.research.google.com/github/brettharder/llm_course/blob/main/01_math_foundations/02_probability_loss_gradients.ipynb) |
| 3 | A decoder transformer from scratch | [Notebook](01_math_foundations/03_decoder_transformer.ipynb) | [Open](https://colab.research.google.com/github/brettharder/llm_course/blob/main/01_math_foundations/03_decoder_transformer.ipynb) |
| 4 | Position encodings and RoPE | [Notebook](01_math_foundations/04_position_encodings_rope.ipynb) | [Open](https://colab.research.google.com/github/brettharder/llm_course/blob/main/01_math_foundations/04_position_encodings_rope.ipynb) |
| 5 | Efficient attention and KV caches | [Notebook](01_math_foundations/05_efficient_attention.ipynb) | [Open](https://colab.research.google.com/github/brettharder/llm_course/blob/main/01_math_foundations/05_efficient_attention.ipynb) |
| 6 | Mixture-of-Experts transformers | [Notebook](01_math_foundations/06_mixture_of_experts.ipynb) | [Open](https://colab.research.google.com/github/brettharder/llm_course/blob/main/01_math_foundations/06_mixture_of_experts.ipynb) |
| 7 | Training data, tokenization, and packing | [Notebook](02_training/07_data_tokenization_packing.ipynb) | [Open](https://colab.research.google.com/github/brettharder/llm_course/blob/main/02_training/07_data_tokenization_packing.ipynb) |
| 8 | Synthetic data pipelines | [Notebook](02_training/08_synthetic_data_pipelines.ipynb) | [Open](https://colab.research.google.com/github/brettharder/llm_course/blob/main/02_training/08_synthetic_data_pipelines.ipynb) |
| 9 | Pretraining a tiny decoder with native PyTorch | [Notebook](02_training/09_native_pytorch_pretraining.ipynb) | [Open](https://colab.research.google.com/github/brettharder/llm_course/blob/main/02_training/09_native_pytorch_pretraining.ipynb) |
| 10 | Pretraining a random-initialized Hugging Face model | [Notebook](02_training/10_hf_random_init_pretraining.ipynb) | [Open](https://colab.research.google.com/github/brettharder/llm_course/blob/main/02_training/10_hf_random_init_pretraining.ipynb) |
| 11 | From base model to instruction, reasoning, and tool use | [Notebook](02_training/11_base_to_posttrained_models.ipynb) | [Open](https://colab.research.google.com/github/brettharder/llm_course/blob/main/02_training/11_base_to_posttrained_models.ipynb) |
| 12 | Optimization and training loops | [Notebook](02_training/12_optimization_training_loop.ipynb) | [Open](https://colab.research.google.com/github/brettharder/llm_course/blob/main/02_training/12_optimization_training_loop.ipynb) |
| 13 | Efficient and distributed training | [Notebook](02_training/13_efficient_distributed_training.ipynb) | [Open](https://colab.research.google.com/github/brettharder/llm_course/blob/main/02_training/13_efficient_distributed_training.ipynb) |
| 14 | SFT with LoRA and QLoRA | [Notebook](02_training/14_sft_lora_qlora.ipynb) | [Open](https://colab.research.google.com/github/brettharder/llm_course/blob/main/02_training/14_sft_lora_qlora.ipynb) |
| 15 | Preference optimization with DPO | [Notebook](02_training/15_preference_optimization_dpo.ipynb) | [Open](https://colab.research.google.com/github/brettharder/llm_course/blob/main/02_training/15_preference_optimization_dpo.ipynb) |
| 16 | Reasoning post-training with GRPO and verifiable rewards | [Notebook](02_training/16_reasoning_grpo_verifiable_rewards.ipynb) | [Open](https://colab.research.google.com/github/brettharder/llm_course/blob/main/02_training/16_reasoning_grpo_verifiable_rewards.ipynb) |
| 17 | Hub, model cards, and inference | [Notebook](03_huggingface/17_hub_models_inference.ipynb) | [Open](https://colab.research.google.com/github/brettharder/llm_course/blob/main/03_huggingface/17_hub_models_inference.ipynb) |
| 18 | Generation, batching, streaming, and structured output | [Notebook](03_huggingface/18_generation_batching_structured.ipynb) | [Open](https://colab.research.google.com/github/brettharder/llm_course/blob/main/03_huggingface/18_generation_batching_structured.ipynb) |
| 19 | Embeddings and semantic search | [Notebook](04_retrieval/19_embeddings_semantic_search.ipynb) | [Open](https://colab.research.google.com/github/brettharder/llm_course/blob/main/04_retrieval/19_embeddings_semantic_search.ipynb) |
| 20 | End-to-end RAG and retrieval evaluation | [Notebook](04_retrieval/20_end_to_end_rag.ipynb) | [Open](https://colab.research.google.com/github/brettharder/llm_course/blob/main/04_retrieval/20_end_to_end_rag.ipynb) |
| 21 | Advanced RAG | [Notebook](04_retrieval/21_advanced_rag.ipynb) | [Open](https://colab.research.google.com/github/brettharder/llm_course/blob/main/04_retrieval/21_advanced_rag.ipynb) |
| 22 | Tool calling and bounded agent loops | [Notebook](05_agents_mcp/22_tool_calling_agents.ipynb) | [Open](https://colab.research.google.com/github/brettharder/llm_course/blob/main/05_agents_mcp/22_tool_calling_agents.ipynb) |
| 23 | Agents with Hugging Face smolagents | [Notebook](05_agents_mcp/23_huggingface_smolagents.ipynb) | [Open](https://colab.research.google.com/github/brettharder/llm_course/blob/main/05_agents_mcp/23_huggingface_smolagents.ipynb) |
| 24 | Building a full MCP server and host bridge | [Notebook](05_agents_mcp/24_mcp_server_hf_client.ipynb) | [Open](https://colab.research.google.com/github/brettharder/llm_course/blob/main/05_agents_mcp/24_mcp_server_hf_client.ipynb) |
| 25 | Durable agent workflows, memory, and human approval | [Notebook](05_agents_mcp/25_durable_agent_workflows_memory.ipynb) | [Open](https://colab.research.google.com/github/brettharder/llm_course/blob/main/05_agents_mcp/25_durable_agent_workflows_memory.ipynb) |
| 26 | Multi-agent workflows and coordination | [Notebook](05_agents_mcp/26_multi_agent_workflows.ipynb) | [Open](https://colab.research.google.com/github/brettharder/llm_course/blob/main/05_agents_mcp/26_multi_agent_workflows.ipynb) |
| 27 | Agent evaluation, sandboxing, and trajectory observability | [Notebook](05_agents_mcp/27_agent_evaluation_sandboxing.ipynb) | [Open](https://colab.research.google.com/github/brettharder/llm_course/blob/main/05_agents_mcp/27_agent_evaluation_sandboxing.ipynb) |
| 28 | Evaluation fundamentals and regression testing | [Notebook](06_evaluation_security/28_evaluation_fundamentals.ipynb) | [Open](https://colab.research.google.com/github/brettharder/llm_course/blob/main/06_evaluation_security/28_evaluation_fundamentals.ipynb) |
| 29 | LLM as a judge | [Notebook](06_evaluation_security/29_llm_as_a_judge.ipynb) | [Open](https://colab.research.google.com/github/brettharder/llm_course/blob/main/06_evaluation_security/29_llm_as_a_judge.ipynb) |
| 30 | LLM application security | [Notebook](06_evaluation_security/30_llm_application_security.ipynb) | [Open](https://colab.research.google.com/github/brettharder/llm_course/blob/main/06_evaluation_security/30_llm_application_security.ipynb) |
| 31 | Experiment tracking, governance, and release engineering | [Notebook](06_evaluation_security/31_experiment_tracking_model_governance.ipynb) | [Open](https://colab.research.google.com/github/brettharder/llm_course/blob/main/06_evaluation_security/31_experiment_tracking_model_governance.ipynb) |
| 32 | Vision-language models and document understanding | [Notebook](07_multimodal/32_vision_language_models.ipynb) | [Open](https://colab.research.google.com/github/brettharder/llm_course/blob/main/07_multimodal/32_vision_language_models.ipynb) |
| 33 | Reliability, observability, and load testing | [Notebook](08_production/33_reliability_observability.ipynb) | [Open](https://colab.research.google.com/github/brettharder/llm_course/blob/main/08_production/33_reliability_observability.ipynb) |
| 34 | Quantization and deployment model formats | [Notebook](08_production/34_quantization_model_formats.ipynb) | [Open](https://colab.research.google.com/github/brettharder/llm_course/blob/main/08_production/34_quantization_model_formats.ipynb) |
| 35 | Choosing an open-model serving engine | [Notebook](08_production/35_serving_engine_comparison.ipynb) | [Open](https://colab.research.google.com/github/brettharder/llm_course/blob/main/08_production/35_serving_engine_comparison.ipynb) |
| 36 | Local model serving with Ollama | [Notebook](08_production/36_ollama_local_serving.ipynb) | [Open](https://colab.research.google.com/github/brettharder/llm_course/blob/main/08_production/36_ollama_local_serving.ipynb) |
| 37 | Serving open models with vLLM | [Notebook](08_production/37_vllm_serving.ipynb) | [Open](https://colab.research.google.com/github/brettharder/llm_course/blob/main/08_production/37_vllm_serving.ipynb) |

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
01_math_foundations/       Decoder, attention, positional, and MoE foundations
02_training/               Data, synthetic data, pretraining, SFT, preferences, and reasoning RL
03_huggingface/            Hub and inference workflows
04_retrieval/              Embeddings, baseline RAG, hybrid retrieval, and reranking
05_agents_mcp/             Tool agents, smolagents, MCP, durable and multi-agent workflows
06_evaluation_security/    Evaluation, judging, security, governance, and release gates
07_multimodal/             Vision-language systems
08_production/             Reliability, quantization, engine selection, Ollama, and vLLM
create_course.py           Notebook generator and core lessons
lesson_expansions.py       Comprehensive lesson/reference expansions
validate_course.py         Static course and security validation
```

## License

No license has been selected yet. Until one is added, normal copyright applies.

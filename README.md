# Rigorous LLM Engineering — Hugging Face First

[![Course validation](https://img.shields.io/badge/notebooks-25%20validated-2ea44f)](#validation)
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
| 1. Mathematical foundations | Tokens and causal modeling · Probability, loss, and gradients · Decoder transformer from scratch · Position encodings and RoPE · Efficient attention, GQA, FlashAttention, and KV caches |
| 2. Training | Data pipelines and packing · Native PyTorch pretraining · Hugging Face random-initialized pretraining · Base-to-post-trained capability pathways · Optimization and distributed training · SFT with LoRA/QLoRA · Preference optimization with DPO |
| 3. Hugging Face inference | Hub, model cards, and inference · Generation, batching, streaming, and structured output |
| 4. Retrieval | Embeddings and semantic search · End-to-end RAG and retrieval evaluation |
| 5. Agents and MCP | Tool calling and agent loops · Building an MCP server |
| 6. Evaluation and security | Evaluation fundamentals · LLM as a judge · LLM application security |
| 7. Multimodal | Vision-language models and document understanding |
| 8. Production | Reliability, observability, and load testing · Serving open models with vLLM |

## Notebook index

Every notebook can be opened directly in Google Colab after the repository is pushed.

| # | Lesson | Notebook | Colab |
|---:|---|---|---|
| 1 | Tokens and causal language modeling | [Notebook](01_math_foundations/01_tokens_and_causal_lm.ipynb) | [Open](https://colab.research.google.com/github/brettharder/llm_course/blob/main/01_math_foundations/01_tokens_and_causal_lm.ipynb) |
| 2 | Probability, cross-entropy, and gradients | [Notebook](01_math_foundations/02_probability_loss_gradients.ipynb) | [Open](https://colab.research.google.com/github/brettharder/llm_course/blob/main/01_math_foundations/02_probability_loss_gradients.ipynb) |
| 3 | A decoder transformer from scratch | [Notebook](01_math_foundations/03_decoder_transformer.ipynb) | [Open](https://colab.research.google.com/github/brettharder/llm_course/blob/main/01_math_foundations/03_decoder_transformer.ipynb) |
| 4 | Position encodings and RoPE | [Notebook](01_math_foundations/04_position_encodings_rope.ipynb) | [Open](https://colab.research.google.com/github/brettharder/llm_course/blob/main/01_math_foundations/04_position_encodings_rope.ipynb) |
| 5 | Efficient attention and KV caches | [Notebook](01_math_foundations/05_efficient_attention.ipynb) | [Open](https://colab.research.google.com/github/brettharder/llm_course/blob/main/01_math_foundations/05_efficient_attention.ipynb) |
| 6 | Training data, tokenization, and packing | [Notebook](02_training/06_data_tokenization_packing.ipynb) | [Open](https://colab.research.google.com/github/brettharder/llm_course/blob/main/02_training/06_data_tokenization_packing.ipynb) |
| 7 | Pretraining a tiny decoder with native PyTorch | [Notebook](02_training/07_native_pytorch_pretraining.ipynb) | [Open](https://colab.research.google.com/github/brettharder/llm_course/blob/main/02_training/07_native_pytorch_pretraining.ipynb) |
| 8 | Pretraining a random-initialized Hugging Face model | [Notebook](02_training/08_hf_random_init_pretraining.ipynb) | [Open](https://colab.research.google.com/github/brettharder/llm_course/blob/main/02_training/08_hf_random_init_pretraining.ipynb) |
| 9 | From base model to instruction, reasoning, and tool use | [Notebook](02_training/09_base_to_posttrained_models.ipynb) | [Open](https://colab.research.google.com/github/brettharder/llm_course/blob/main/02_training/09_base_to_posttrained_models.ipynb) |
| 10 | Optimization and training loops | [Notebook](02_training/10_optimization_training_loop.ipynb) | [Open](https://colab.research.google.com/github/brettharder/llm_course/blob/main/02_training/10_optimization_training_loop.ipynb) |
| 11 | Efficient and distributed training | [Notebook](02_training/11_efficient_distributed_training.ipynb) | [Open](https://colab.research.google.com/github/brettharder/llm_course/blob/main/02_training/11_efficient_distributed_training.ipynb) |
| 12 | SFT with LoRA and QLoRA | [Notebook](02_training/12_sft_lora_qlora.ipynb) | [Open](https://colab.research.google.com/github/brettharder/llm_course/blob/main/02_training/12_sft_lora_qlora.ipynb) |
| 13 | Preference optimization with DPO | [Notebook](02_training/13_preference_optimization_dpo.ipynb) | [Open](https://colab.research.google.com/github/brettharder/llm_course/blob/main/02_training/13_preference_optimization_dpo.ipynb) |
| 14 | Hub, model cards, and inference | [Notebook](03_huggingface/14_hub_models_inference.ipynb) | [Open](https://colab.research.google.com/github/brettharder/llm_course/blob/main/03_huggingface/14_hub_models_inference.ipynb) |
| 15 | Generation, batching, streaming, and structured output | [Notebook](03_huggingface/15_generation_batching_structured.ipynb) | [Open](https://colab.research.google.com/github/brettharder/llm_course/blob/main/03_huggingface/15_generation_batching_structured.ipynb) |
| 16 | Embeddings and semantic search | [Notebook](04_retrieval/16_embeddings_semantic_search.ipynb) | [Open](https://colab.research.google.com/github/brettharder/llm_course/blob/main/04_retrieval/16_embeddings_semantic_search.ipynb) |
| 17 | End-to-end RAG and retrieval evaluation | [Notebook](04_retrieval/17_end_to_end_rag.ipynb) | [Open](https://colab.research.google.com/github/brettharder/llm_course/blob/main/04_retrieval/17_end_to_end_rag.ipynb) |
| 18 | Tool calling and bounded agent loops | [Notebook](05_agents_mcp/18_tool_calling_agents.ipynb) | [Open](https://colab.research.google.com/github/brettharder/llm_course/blob/main/05_agents_mcp/18_tool_calling_agents.ipynb) |
| 19 | Building an MCP server | [Notebook](05_agents_mcp/19_mcp_server_hf_client.ipynb) | [Open](https://colab.research.google.com/github/brettharder/llm_course/blob/main/05_agents_mcp/19_mcp_server_hf_client.ipynb) |
| 20 | Evaluation fundamentals and regression testing | [Notebook](06_evaluation_security/20_evaluation_fundamentals.ipynb) | [Open](https://colab.research.google.com/github/brettharder/llm_course/blob/main/06_evaluation_security/20_evaluation_fundamentals.ipynb) |
| 21 | LLM as a judge | [Notebook](06_evaluation_security/21_llm_as_a_judge.ipynb) | [Open](https://colab.research.google.com/github/brettharder/llm_course/blob/main/06_evaluation_security/21_llm_as_a_judge.ipynb) |
| 22 | LLM application security | [Notebook](06_evaluation_security/22_llm_application_security.ipynb) | [Open](https://colab.research.google.com/github/brettharder/llm_course/blob/main/06_evaluation_security/22_llm_application_security.ipynb) |
| 23 | Vision-language models and document understanding | [Notebook](07_multimodal/23_vision_language_models.ipynb) | [Open](https://colab.research.google.com/github/brettharder/llm_course/blob/main/07_multimodal/23_vision_language_models.ipynb) |
| 24 | Reliability, observability, and load testing | [Notebook](08_production/24_reliability_observability.ipynb) | [Open](https://colab.research.google.com/github/brettharder/llm_course/blob/main/08_production/24_reliability_observability.ipynb) |
| 25 | Serving open models with vLLM | [Notebook](08_production/25_vllm_serving.ipynb) | [Open](https://colab.research.google.com/github/brettharder/llm_course/blob/main/08_production/25_vllm_serving.ipynb) |

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
01_math_foundations/       Decoder and mathematical foundations
02_training/               Data, pretraining, post-training, optimization, SFT, and preferences
03_huggingface/            Hub and inference workflows
04_retrieval/              Embeddings and RAG
05_agents_mcp/             Tool agents and MCP
06_evaluation_security/    Evaluation, judging, and security
07_multimodal/             Vision-language systems
08_production/             Reliability and vLLM serving
create_course.py           Notebook generator and core lessons
lesson_expansions.py       Comprehensive lesson/reference expansions
validate_course.py         Static course and security validation
```

## License

No license has been selected yet. Until one is added, normal copyright applies.

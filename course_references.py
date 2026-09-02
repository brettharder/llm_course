"""Curated primary papers, specifications, and official documentation by lesson."""

from __future__ import annotations


REFERENCES = {
    1: [("NumPy broadcasting", "https://numpy.org/doc/stable/user/basics.broadcasting.html"), ("NumPy einsum", "https://numpy.org/doc/stable/reference/generated/numpy.einsum.html")],
    2: [("PyTorch tensor views", "https://docs.pytorch.org/docs/stable/tensor_view.html"), ("PyTorch autograd mechanics", "https://docs.pytorch.org/docs/stable/notes/autograd.html"), ("PyTorch data utilities", "https://docs.pytorch.org/docs/stable/data.html")],
    3: [("PyTorch numerical accuracy", "https://docs.pytorch.org/docs/stable/notes/numerical_accuracy.html"), ("PyTorch reproducibility", "https://docs.pytorch.org/docs/stable/notes/randomness.html")],
    4: [("Neural probabilistic language model", "https://www.jmlr.org/papers/v3/bengio03a.html"), ("SentencePiece", "https://arxiv.org/abs/1808.06226")],
    5: [("PyTorch cross entropy", "https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html"), ("Deep Learning: numerical computation", "https://www.deeplearningbook.org/contents/numerical.html")],
    6: [("Hugging Face Tokenizers", "https://huggingface.co/docs/tokenizers/index"), ("Tokenizer summary", "https://huggingface.co/docs/transformers/tokenizer_summary")],
    7: [("Attention Is All You Need", "https://arxiv.org/abs/1706.03762"), ("Language Models are Unsupervised Multitask Learners", "https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf")],
    8: [("RoFormer / RoPE", "https://arxiv.org/abs/2104.09864"), ("YaRN", "https://arxiv.org/abs/2309.00071")],
    9: [("FlashAttention", "https://arxiv.org/abs/2205.14135"), ("From Multi-Head to Grouped-Query Attention", "https://arxiv.org/abs/2305.13245"), ("PyTorch scaled dot-product attention", "https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html")],
    10: [("Switch Transformers", "https://arxiv.org/abs/2101.03961"), ("Mixture-of-Experts with Expert Choice Routing", "https://arxiv.org/abs/2202.09368")],
    11: [("Scaling Laws for Neural Language Models", "https://arxiv.org/abs/2001.08361"), ("Training Compute-Optimal Large Language Models", "https://arxiv.org/abs/2203.15556")],
    12: [("A Mathematical Framework for Transformer Circuits", "https://transformer-circuits.pub/2021/framework/index.html"), ("Tracing the mechanisms of language model factual recall", "https://arxiv.org/abs/2202.05262")],
    13: [("Hugging Face Datasets", "https://huggingface.co/docs/datasets/index"), ("Deduplicating Training Data Makes Language Models Better", "https://arxiv.org/abs/2107.06499")],
    14: [("Self-Instruct", "https://arxiv.org/abs/2212.10560"), ("Textbooks Are All You Need", "https://arxiv.org/abs/2306.11644")],
    15: [("PyTorch Transformer building blocks", "https://docs.pytorch.org/tutorials/intermediate/transformer_building_blocks.html"), ("AdamW", "https://arxiv.org/abs/1711.05101")],
    16: [("Transformers model creation", "https://huggingface.co/docs/transformers/create_a_model"), ("Transformers Trainer", "https://huggingface.co/docs/transformers/trainer")],
    17: [("Training language models to follow instructions", "https://arxiv.org/abs/2203.02155"), ("TRL documentation", "https://huggingface.co/docs/trl/index")],
    18: [("AdamW", "https://arxiv.org/abs/1711.05101"), ("PyTorch automatic mixed precision", "https://docs.pytorch.org/docs/stable/amp.html")],
    19: [("PyTorch FSDP", "https://docs.pytorch.org/docs/stable/fsdp.html"), ("PyTorch activation checkpointing", "https://docs.pytorch.org/docs/stable/checkpoint.html"), ("ZeRO", "https://arxiv.org/abs/1910.02054")],
    20: [("LoRA", "https://arxiv.org/abs/2106.09685"), ("QLoRA", "https://arxiv.org/abs/2305.14314"), ("PEFT documentation", "https://huggingface.co/docs/peft/index")],
    21: [("Direct Preference Optimization", "https://arxiv.org/abs/2305.18290"), ("TRL DPOTrainer", "https://huggingface.co/docs/trl/dpo_trainer")],
    22: [("DeepSeekMath / GRPO", "https://arxiv.org/abs/2402.03300"), ("TRL GRPOTrainer", "https://huggingface.co/docs/trl/grpo_trainer")],
    23: [("Distilling the Knowledge in a Neural Network", "https://arxiv.org/abs/1503.02531"), ("Generalized Knowledge Distillation", "https://arxiv.org/abs/2306.13649"), ("TRL trainers", "https://huggingface.co/docs/trl/main/trainer")],
    24: [("Hugging Face Hub documentation", "https://huggingface.co/docs/hub/index"), ("Transformers custom models and security", "https://huggingface.co/docs/transformers/custom_models")],
    25: [("Transformers chat templates", "https://huggingface.co/docs/transformers/chat_templating"), ("Writing chat templates", "https://huggingface.co/docs/transformers/chat_templating_writing")],
    26: [("Transformers pipelines", "https://huggingface.co/docs/transformers/main_classes/pipelines"), ("Hugging Face inference providers", "https://huggingface.co/docs/inference-providers/index")],
    27: [("Transformers generation", "https://huggingface.co/docs/transformers/main_classes/text_generation"), ("Generation strategies", "https://huggingface.co/docs/transformers/generation_strategies")],
    28: [("Sentence-BERT", "https://arxiv.org/abs/1908.10084"), ("FAISS", "https://faiss.ai/")],
    29: [("Sentence Transformers training", "https://www.sbert.net/docs/sentence_transformer/training_overview.html"), ("Sentence Transformers CrossEncoder", "https://www.sbert.net/docs/cross_encoder/usage/usage.html")],
    30: [("Retrieval-Augmented Generation", "https://arxiv.org/abs/2005.11401"), ("BEIR", "https://arxiv.org/abs/2104.08663")],
    31: [("Reciprocal Rank Fusion", "https://doi.org/10.1145/1571941.1572114"), ("HyDE", "https://arxiv.org/abs/2212.10496")],
    32: [("ReAct", "https://arxiv.org/abs/2210.03629"), ("Toolformer", "https://arxiv.org/abs/2302.04761")],
    33: [("Hugging Face smolagents", "https://huggingface.co/docs/smolagents/index"), ("smolagents guided tour", "https://huggingface.co/docs/smolagents/guided_tour")],
    34: [("Model Context Protocol specification", "https://modelcontextprotocol.io/specification/latest"), ("MCP security best practices", "https://modelcontextprotocol.io/specification/latest/basic/security_best_practices")],
    35: [("Temporal durable execution", "https://docs.temporal.io/temporal"), ("LangGraph durable execution", "https://docs.langchain.com/oss/python/langgraph/durable-execution")],
    36: [("AutoGen", "https://arxiv.org/abs/2308.08155"), ("CAMEL", "https://arxiv.org/abs/2303.17760")],
    37: [("AgentBench", "https://arxiv.org/abs/2308.03688"), ("NIST AI Risk Management Framework", "https://www.nist.gov/itl/ai-risk-management-framework")],
    38: [("Hugging Face Evaluate", "https://huggingface.co/docs/evaluate/index"), ("Bootstrap methods", "https://doi.org/10.1214/aos/1176344552")],
    39: [("Judging LLM-as-a-Judge", "https://arxiv.org/abs/2306.05685"), ("G-Eval", "https://arxiv.org/abs/2303.16634")],
    40: [("LightEval documentation", "https://huggingface.co/docs/lighteval/index"), ("LightEval Python API", "https://huggingface.co/docs/lighteval/using-the-python-api"), ("LightEval task inspection", "https://huggingface.co/docs/lighteval/available-tasks")],
    41: [("OWASP Top 10 for LLM Applications", "https://genai.owasp.org/llm-top-10/"), ("NIST AI Risk Management Framework", "https://www.nist.gov/itl/ai-risk-management-framework")],
    42: [("Model Cards", "https://arxiv.org/abs/1810.03993"), ("Datasheets for Datasets", "https://arxiv.org/abs/1803.09010")],
    43: [("CLIP", "https://arxiv.org/abs/2103.00020"), ("LLaVA", "https://arxiv.org/abs/2304.08485"), ("Transformers multimodal chat templates", "https://huggingface.co/docs/transformers/chat_templating_multimodal")],
    44: [("Google SRE book", "https://sre.google/sre-book/table-of-contents/"), ("OpenTelemetry documentation", "https://opentelemetry.io/docs/")],
    45: [("GPTQ", "https://arxiv.org/abs/2210.17323"), ("AWQ", "https://arxiv.org/abs/2306.00978"), ("Transformers quantization", "https://huggingface.co/docs/transformers/quantization/overview")],
    46: [("llama.cpp", "https://github.com/ggml-org/llama.cpp"), ("SGLang documentation", "https://docs.sglang.ai/"), ("vLLM documentation", "https://docs.vllm.ai/")],
    47: [("Ollama documentation", "https://docs.ollama.com/"), ("Ollama Modelfile reference", "https://docs.ollama.com/modelfile")],
    48: [("vLLM / PagedAttention", "https://arxiv.org/abs/2309.06180"), ("vLLM documentation", "https://docs.vllm.ai/"), ("vLLM metrics", "https://docs.vllm.ai/en/latest/design/metrics/")],
    49: [("SentencePiece", "https://arxiv.org/abs/1808.06226"), ("Attention Is All You Need", "https://arxiv.org/abs/1706.03762"), ("Model Cards", "https://arxiv.org/abs/1810.03993")],
    50: [("Retrieval-Augmented Generation", "https://arxiv.org/abs/2005.11401"), ("BEIR", "https://arxiv.org/abs/2104.08663"), ("OWASP LLM guidance", "https://genai.owasp.org/llm-top-10/")],
    51: [("ReAct", "https://arxiv.org/abs/2210.03629"), ("Model Context Protocol", "https://modelcontextprotocol.io/specification/latest"), ("NIST AI RMF", "https://www.nist.gov/itl/ai-risk-management-framework")],
}


def register(lessons, md) -> None:
    by_number = {int(path.rsplit("/", 1)[1].split("_", 1)[0]): cells for path, _, cells in lessons}
    if set(REFERENCES) != set(by_number):
        raise ValueError("Every lesson must have a curated reference set")
    for number, references in REFERENCES.items():
        links = "\n".join(f"- [{title}]({url})" for title, url in references)
        by_number[number][-1:-1] = [md(
            "## Primary references and further study\n\n"
            "Use the pinned library documentation that matches your environment. Papers explain the "
            "method and assumptions; current official documentation defines the executable API.\n\n" + links
        )]

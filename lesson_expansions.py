"""Deep lesson/reference material injected into each generated notebook."""

EXPANSIONS: dict[int, list[tuple[str, str]]] = {}


def lesson(number: int, *cells: tuple[str, str]) -> None:
    EXPANSIONS[number] = list(cells)


lesson(1,
    ("md", r"""
    ## 1.4 How modern subword tokenizers are built

    A useful tokenizer must be lossless, reasonably compact, fast, and stable across the
    model's target languages and domains. Byte-level BPE begins with bytes, so every input is
    representable. It repeatedly merges frequent adjacent symbols until it reaches a target
    vocabulary size. WordPiece uses a related greedy vocabulary construction. Unigram begins
    with a large candidate vocabulary and removes pieces while minimizing a probabilistic
    objective. SentencePiece treats whitespace as an ordinary symbol and can train directly
    on raw text.

    Vocabulary construction is a distributional decision. A tokenizer trained mainly on
    English prose may split another language, source code, chemical notation, or identifiers
    inefficiently. That increases sequence length, attention cost, and the number of prediction
    steps. Special tokens also carry architectural meaning: BOS/EOS delimit sequences; PAD
    aligns batches; chat-control tokens mark roles; tool or multimodal tokens may reserve spans.
    Adding tokens after pretraining creates randomly initialized embedding/output rows unless
    they are deliberately initialized and trained.

    **Reference rule:** never infer token counts from character counts in production. Use the
    exact tokenizer revision paired with the exact model revision and chat template.
    """),
    ("code", r'''
    # Inspect vocabulary behavior and round-trip invariants.
    probes = [
        "hello", " hello", "hello\n", "HTTPResponseFactory", "2.718281828",
        "naïve café", "مرحبا بالعالم", "中文分词", "👩🏽‍💻",
    ]
    for text in probes:
        ids = tokenizer.encode(text, add_special_tokens=False)
        decoded = tokenizer.decode(ids)
        print({"text": text, "tokens": len(ids), "ids": ids,
               "round_trip": decoded == text, "decoded": decoded})

    special = tokenizer.special_tokens_map
    print("\nspecial tokens:", special)
    print("vocabulary size:", tokenizer.vocab_size)
    '''),
    ("md", r"""
    ## 1.5 Sequence construction, boundaries, and label masking

    Language-model examples are usually concatenated and sliced into fixed-length blocks.
    Boundaries matter. Without EOS separators, the model is trained to continue the end of one
    document with the beginning of an unrelated one. Packing improves token utilization but
    can allow cross-example attention unless a block-diagonal mask is used. During instruction
    tuning, training on every role teaches the model to reproduce user text as well as assistant
    text; completion-only or assistant-only masks instead put `-100` on non-target positions.

    Padding and loss masking solve different problems. The attention mask says which positions
    may participate in attention. A `-100` label tells PyTorch cross-entropy not to score that
    position. Decoder-only generation is commonly left-padded in batches because generation
    begins after the final array position, while training is often right-padded. Inspect rather
    than assume a model's pad token: many causal models reuse EOS, which is acceptable only when
    masks correctly distinguish padding from actual EOS occurrences.
    """),
    ("code", r'''
    # A fully explicit next-token example.
    text = "Attention reuses cached keys."
    ids = tokenizer.encode(text, add_special_tokens=False)
    print("position | input piece -> target piece")
    for position, (current, target) in enumerate(zip(ids[:-1], ids[1:])):
        print(f"{position:8d} | {tokenizer.decode([current])!r:15} -> {tokenizer.decode([target])!r}")

    # Each row of a causal mask can see itself and earlier positions only.
    import torch
    T = min(len(ids), 8)
    allowed = torch.tril(torch.ones(T, T, dtype=torch.int))
    print("\ncausal visibility mask (query rows, key columns):\n", allowed)
    '''),
    ("md", r"""
    ## 1.6 Context windows and practical token budgets

    A context limit covers prompt, chat-control tokens, retrieved evidence, tool schemas and
    results, prior messages, and generated output. Reserving no output budget is a common bug.
    The maximum advertised window is also not a promise of equal accuracy at every position.
    Long contexts increase prefill work and KV-cache memory, may dilute relevant evidence, and
    can suffer position-dependent retrieval failures.

    Build token budgeting as a deterministic preprocessing stage: render the final chat
    template, count exact tokens, reserve output and safety margin, then apply an explicit
    policy—truncate low-priority history, summarize, retrieve fewer chunks, or reject. Silent
    truncation can remove the user's question, an assistant target, citations, or image tokens.

    **Diagnostic checklist:** record tokenizer ID/revision, rendered prompt token count, special
    tokens added, truncation side, content removed, reserved output, and model context limit.
    Tokenization bugs often masquerade as model-quality problems.
    """),
)

lesson(2,
    ("md", r"""
    ## 2.4 Log-sum-exp, token reduction, and masking

    Stable cross-entropy is normally implemented without constructing probabilities explicitly:

    \[
    \log\sum_j e^{z_j}=m+\log\sum_j e^{z_j-m},\quad m=\max_j z_j.
    \]

    The per-token negative log-likelihood is `logsumexp(logits) - target_logit`. Framework
    functions fuse these operations for stability and speed. Reduction deserves attention.
    A mean over non-padding tokens weights long examples more heavily than short examples. A
    mean of per-example means weights examples equally. Neither is universally correct; choose
    the unit implied by the task and report it.

    Masked loss is not equivalent to deleting input. Masked prompt tokens still condition the
    assistant response; they simply receive no direct target loss. Label smoothing replaces a
    one-hot target with a mixture containing a small uniform component, discouraging extreme
    confidence but potentially harming exact generation. Class weights address imbalance in
    classification, while token-level language modeling usually needs data sampling or explicit
    token/example weighting instead.
    """),
    ("code", r'''
    # Reconstruct cross-entropy from log-sum-exp and compare reductions.
    import torch
    logits = torch.tensor([
        [[3., 1., 0.], [0., 2., 1.], [1., 0., 3.]],
        [[2., 0., 1.], [1., 3., 0.], [0., 0., 0.]],
    ])
    labels = torch.tensor([[0, 1, 2], [2, 1, -100]])
    safe = labels.clamp_min(0)
    target_logits = logits.gather(-1, safe.unsqueeze(-1)).squeeze(-1)
    nll = torch.logsumexp(logits, -1) - target_logits
    valid = labels.ne(-100)
    token_mean = nll[valid].mean()
    example_means = [(row[mask]).mean() for row, mask in zip(nll, valid)]
    example_mean = torch.stack(example_means).mean()
    builtin = torch.nn.functional.cross_entropy(logits.flatten(0, 1), labels.flatten())
    print("token mean:", token_mean.item(), "builtin:", builtin.item())
    print("equal-example mean:", example_mean.item())
    '''),
    ("md", r"""
    ## 2.5 Backpropagation beyond the final logits

    The elegant gradient `p - y` is only the start. The chain rule propagates it through the
    LM head, residual stream, attention, MLPs, embeddings, and earlier tokens. Shared/tied input
    embeddings and output weights receive gradient from both roles. Residual connections sum
    gradient paths. Normalization changes scale and coupling. Attention routes gradients across
    visible prior positions, while the causal mask blocks future paths.

    Gradient magnitude alone is not parameter importance. Adaptive optimizers rescale updates
    using running moments; weight decay adds a separate shrinkage; mixed precision may scale the
    loss before backward. For diagnosis, track global and per-module gradient norms, zero or NaN
    fractions, update-to-weight ratios, and activation statistics. Clip after unscaling and before
    the optimizer step. Persistent clipping means the learning rate, data, or loss scale deserves
    investigation—not that clipping has “fixed” training.
    """),
    ("code", r'''
    # Watch a tiny model learn and inspect gradient/update scales.
    torch.manual_seed(7)
    layer = torch.nn.Linear(4, 3)
    opt = torch.optim.SGD(layer.parameters(), lr=0.2)
    x = torch.tensor([[1., 0., 1., 0.], [0., 1., 0., 1.]])
    y = torch.tensor([0, 2])
    for step in range(6):
        before = layer.weight.detach().clone()
        opt.zero_grad(set_to_none=True)
        loss = torch.nn.functional.cross_entropy(layer(x), y)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(layer.parameters(), 10.0)
        opt.step()
        update = (layer.weight.detach() - before).norm()
        print(step, f"loss={loss.item():.4f}", f"grad={grad_norm:.4f}", f"update={update:.4f}")
    '''),
    ("md", r"""
    ## 2.6 Perplexity, entropy, and calibration reference

    Perplexity exponentiates average NLL. It is sensitive to tokenizer, domain, sequence
    boundary policy, masking, and reduction. It is useful for comparing checkpoints under one
    controlled protocol, but a lower perplexity does not guarantee better instruction following,
    truthfulness, safety, or downstream utility. Bits per byte/character can make comparisons
    across tokenizers fairer by normalizing to the underlying representation.

    Entropy describes uncertainty in the model distribution; cross-entropy measures how that
    distribution scores observed targets; KL divergence measures distribution discrepancy.
    Calibration asks whether events predicted with probability 0.8 happen about 80% of the time.
    Generative calibration is difficult because there are many valid sequences. For constrained
    labels, use reliability diagrams, expected calibration error, Brier score, and temperature
    scaling on held-out data. Never tune calibration on the final test set.
    """),
)

lesson(3,
    ("md", r"""
    ## 3.3 Complete decoder data flow

    A decoder-only transformer performs: token lookup → positional transformation → repeated
    decoder blocks → final normalization → vocabulary projection. With tied embeddings, the
    output matrix is the transpose of the input embedding matrix, reducing parameters and
    encouraging a shared lexical geometry. Each block contains a communication operation
    (causal attention) and a per-position computation operation (MLP).

    Modern MLPs often use gated activations such as SwiGLU rather than a two-layer GELU MLP:
    `down(silu(gate(x)) * up(x))`. Gating increases projection parameters but often improves
    quality. Biases may be omitted. RMSNorm is common. Dropout is frequently zero in large-model
    pretraining. Architectural labels such as “Llama-like” still hide important choices: head
    counts, GQA groups, RoPE base/scaling, vocabulary, tie policy, activation, norm epsilon,
    initialization, and local/sliding attention.
    """),
    ("code", r'''
    # Assemble a tiny causal LM from the previously defined block.
    class TinyDecoderLM(nn.Module):
        def __init__(self, vocab=128, d_model=64, layers=3, heads=4):
            super().__init__()
            self.embed = nn.Embedding(vocab, d_model)
            self.blocks = nn.ModuleList([DecoderBlock(d_model, heads) for _ in range(layers)])
            self.final_norm = RMSNorm(d_model)
            self.lm_head = nn.Linear(d_model, vocab, bias=False)
            self.lm_head.weight = self.embed.weight  # tied weights
        def forward(self, ids):
            x = self.embed(ids)
            maps = []
            for block in self.blocks:
                x, attention = block(x); maps.append(attention)
            return self.lm_head(self.final_norm(x)), maps

    tiny = TinyDecoderLM()
    ids = torch.randint(0, 128, (2, 12))
    logits, maps = tiny(ids)
    labels = ids.clone()
    loss = nn.functional.cross_entropy(logits[:, :-1].reshape(-1, 128), labels[:, 1:].reshape(-1))
    loss.backward()
    print("logits", logits.shape, "loss", loss.item())
    print("tied storage:", tiny.lm_head.weight.data_ptr() == tiny.embed.weight.data_ptr())
    '''),
    ("md", r"""
    ## 3.4 Parameter and compute accounting

    Ignoring biases, standard attention projections contain roughly \(4D^2\) parameters
    (Q, K, V, output). A conventional 4× MLP contains roughly \(8D^2\). Thus one block is
    about \(12D^2\), before embeddings and norms. GQA reduces K/V projection parameters but
    not Q/output. A gated MLP with two input projections and one output projection changes the
    count. Embeddings cost \(VD\), which is substantial for large vocabularies and small models.

    Training compute is often summarized as approximately six times parameter count times
    training tokens for dense transformers, but this is a planning approximation. Actual FLOPs
    depend on sequence length, attention, activation checkpointing, sparsity/MoE routing, and
    implementation. Inference separates prefill (parallel prompt processing) from decode
    (sequential, cache-reading token steps). Parameter count alone does not predict latency.
    """),
    ("code", r'''
    def rough_decoder_params(vocab, width, layers, mlp_ratio=4, tied=True):
        embeddings = vocab * width * (1 if tied else 2)
        attention = layers * 4 * width**2
        mlp = layers * 2 * mlp_ratio * width**2
        norms = layers * 2 * width + width
        return {"embeddings": embeddings, "attention": attention, "mlp": mlp,
                "norms": norms, "total": embeddings + attention + mlp + norms}

    for config in [(32_000, 768, 12), (128_000, 2048, 24), (128_000, 4096, 32)]:
        parts = rough_decoder_params(*config)
        print(config, {k: f"{v/1e9:.3f}B" for k, v in parts.items()})
    '''),
    ("md", r"""
    ## 3.5 Initialization, residual scale, and architecture diagnostics

    If activations or residual updates grow with depth, softmax and nonlinearities can saturate
    and gradients can destabilize. Initialization scales projection weights; some architectures
    scale residual output projections by depth. Pre-norm provides a clean identity path but can
    produce large residual streams; post-norm changes optimization behavior. Deep networks also
    benefit from careful optimizer warmup and precision choices.

    When implementing a block, test invariants before training: shapes for multiple batch/lengths;
    no attention above the causal diagonal; no NaNs on extreme but valid inputs; deterministic
    forward under eval; gradients reach every intended parameter; padding does not influence
    non-padding outputs; cached and uncached generation agree; and a tiny dataset can be
    overfit. The “overfit one batch” test is one of the fastest ways to expose target shifting,
    masks, detached tensors, or optimizer errors.
    """),
)

lesson(4,
    ("md", r"""
    ## 4.4 Comparing positional strategies

    Learned absolute embeddings are simple and flexible inside their trained range, but have a
    fixed table and poor extrapolation. Sinusoidal encodings require no learned table and expose
    multiple wavelengths, yet add position to token state rather than attention relations.
    Relative position biases directly modify attention scores by displacement buckets. ALiBi
    uses a monotonic, head-specific distance penalty and extrapolates operationally without a
    table. RoPE rotates Q/K features and has become common in decoder LLMs. Some architectures
    mix sliding/local attention with periodic global layers, changing what “position handling”
    means at long range.

    There is no context extension switch independent of training. Interpolation compresses new
    positions into the trained range; NTK-aware and frequency-selective variants adjust RoPE
    frequencies; YaRN-like approaches combine scaling and attention adjustments. Each changes
    the distribution the model sees. Long-context continued training may be required, and
    evaluation must test more than a single retrieval needle.
    """),
    ("code", r'''
    # Visualize RoPE wavelengths and rotations across positions.
    import matplotlib.pyplot as plt
    D = 16
    inv_freq = 10_000 ** (-torch.arange(0, D, 2) / D)
    positions = torch.arange(0, 256)
    angles = positions[:, None] * inv_freq[None, :]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.5))
    axes[0].plot(inv_freq.numpy(), marker="o")
    axes[0].set(title="RoPE inverse frequencies", xlabel="feature pair", ylabel="radians/position")
    for i in [0, 2, 4, 7]: axes[1].plot(positions, angles[:, i].cos(), label=f"pair {i}")
    axes[1].set(title="Cosine phase by position", xlabel="position"); axes[1].legend()
    plt.tight_layout()
    '''),
    ("md", r"""
    ## 4.5 Relative-position identity and implementation details

    Rotation matrices obey \(R(a)^T R(b)=R(b-a)\). Therefore
    \((R(m)q)^T(R(n)k)=q^T R(n-m)k\): the attention dot product exposes relative displacement
    while each vector also carries absolute phase. Implementations often use a “rotate half”
    arrangement rather than adjacent feature pairs; both are valid only when frequencies and
    layout match model training. RoPE is applied after Q/K projection, commonly to only a rotary
    sub-dimension, and must use absolute cache positions during incremental decoding.

    Cache correctness is a frequent failure: if a new token is rotated as position zero rather
    than its actual sequence index, cached generation diverges. Padding requires explicit
    position IDs in some batching layouts. Packed sequences may reset positions per example or
    continue monotonically, depending on training design. Changing base, scaling, rotary
    percentage, or layout makes existing weights incompatible even though tensor shapes match.
    """),
    ("code", r'''
    # Numerically verify that jointly shifting q/k positions preserves their RoPE dot product.
    torch.manual_seed(3)
    q, k = torch.randn(1, 1, 8), torch.randn(1, 1, 8)
    def score(q_pos, k_pos):
        qr = rope(q, torch.tensor([q_pos]))
        kr = rope(k, torch.tensor([k_pos]))
        return float((qr * kr).sum())
    for shift in [0, 5, 100]:
        print(shift, score(7 + shift, 13 + shift))
    print("different displacement:", score(7, 14))
    '''),
    ("md", r"""
    ## 4.6 Long-context evaluation reference

    Test several capabilities and positions: exact retrieval, multi-hop synthesis across distant
    passages, aggregation over many records, instruction persistence, conflicting evidence,
    recent versus early evidence, and generation after a long prefill. Include distractors that
    share vocabulary with the question. Measure accuracy by depth and total length, TTFT,
    memory, and tokens/second. Inspect whether failures come from truncation, retrieval,
    attention, generation, or the evaluation itself.

    “Needle in a haystack” is a diagnostic, not a complete benchmark: exact distinctive strings
    can be matched without robust comprehension. Long-context RAG may still outperform placing
    everything in context because retrieval filters distraction and reduces compute. Conversely,
    retrieval can omit decisive evidence. Choose architecture using representative tasks and
    end-to-end cost, not maximum context length on a model card.
    """),
)

lesson(5,
    ("md", r"""
    ## 5.4 Attention variants in shape notation

    Let query heads be \(H_q\), KV heads \(H_{kv}\), and group size
    \(g=H_q/H_{kv}\). MHA has \(H_{kv}=H_q\), MQA has \(H_{kv}=1\), and GQA lies between.
    During attention, each K/V head is logically shared by g query heads. The Q projection
    remains \(D\times D\); K and V projections shrink in proportion to KV heads. Output shape
    remains `[B,T,D]`, so downstream blocks are unchanged.

    Other efficiency families solve different problems. Sliding-window attention restricts each
    token to a local neighborhood, reducing long-sequence work but limiting direct interaction.
    Block-sparse attention chooses structured connections. Linear-attention methods replace or
    reorder softmax attention using kernel/state formulations and are generally approximate or
    architecturally different. Mixture-of-experts sparsifies MLP parameter activation, not
    attention. Do not group all of these under “FlashAttention”: Flash changes execution of the
    same dense attention result.
    """),
    ("code", r'''
    # Compare projection parameters and cache bytes for MHA/GQA/MQA.
    def attention_accounting(width, q_heads, kv_heads, layers, tokens, dtype_bytes=2):
        head_dim = width // q_heads
        q = width * width
        k_and_v = 2 * width * (kv_heads * head_dim)
        out = width * width
        cache = 2 * layers * kv_heads * head_dim * tokens * dtype_bytes
        return q + k_and_v + out, cache

    for label, kv in [("MHA", 32), ("GQA-8", 8), ("GQA-4", 4), ("MQA", 1)]:
        params, cache = attention_accounting(4096, 32, kv, 32, 32_768)
        print(f"{label:6} projections={params/1e6:7.1f}M  cache={cache/2**30:5.2f} GiB")
    '''),
    ("md", r"""
    ## 5.5 Online softmax: why tiling can be exact

    Softmax seems to require an entire row because its denominator sums all keys. Online
    softmax processes blocks while maintaining a running maximum \(m\) and normalized sum
    \(l\). When a new block has a larger maximum, earlier accumulators are rescaled by
    \(e^{m_{old}-m_{new}}\). The output accumulator is updated with the same correction. This
    lets a kernel tile Q/K/V in fast on-chip memory without storing the full score matrix in
    device memory. Backward recomputes inexpensive intermediates rather than reading huge saved
    matrices.

    Kernel dispatch has constraints: device generation, dtype, head dimension, mask type,
    dropout, contiguity, and library version. PyTorch scaled-dot-product attention selects among
    Flash, memory-efficient, and math backends when eligible. A fallback is correct but can have
    very different memory/performance. Benchmark after warmup with synchronization, realistic
    shapes, and peak-memory measurement.
    """),
    ("code", r'''
    # Simple backend-agnostic attention benchmark scaffold.
    import time
    def benchmark_sdpa(B=2, H=8, T=512, Dh=64, repeats=10, device="cpu"):
        q = torch.randn(B, H, T, Dh, device=device)
        for _ in range(2):
            torch.nn.functional.scaled_dot_product_attention(q, q, q, is_causal=True)
        if device == "cuda": torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(repeats):
            torch.nn.functional.scaled_dot_product_attention(q, q, q, is_causal=True)
        if device == "cuda": torch.cuda.synchronize()
        return (time.perf_counter() - start) / repeats

    device = "cuda" if torch.cuda.is_available() else "cpu"
    for T in [128, 256, 512]:
        print(T, f"{benchmark_sdpa(T=T, repeats=3, device=device)*1000:.2f} ms")
    '''),
    ("md", r"""
    ## 5.6 KV-cache operations reference

    The cache stores per-layer K/V after positional transformation. Dynamic caches grow with
    tokens and are convenient; static caches preallocate a maximum shape and can work better
    with compilation but may waste space. Sliding caches retain a fixed recent window. Offloaded
    caches move data across slower links. Quantized caches reduce bytes but add conversion and
    may affect quality. Prefix caching reuses identical prompt blocks across requests; it needs
    canonical token sequences and cache-aware scheduling.

    Continuous batching interleaves decode steps for active sequences. Paged allocation maps
    logical token blocks to physical cache blocks, reducing fragmentation and enabling sharing.
    Capacity planning must include batch/concurrency and total cached tokens, not only maximum
    context per request. Monitor cache utilization, evictions, prefix hit rate, prefill/decode
    throughput, queue time, TTFT, and inter-token latency. OOM under load is an admission-control
    failure even if one maximum-length request fits in isolation.
    """),
)

lesson(7,
    ("md", r"""
    ## 7.3 Dataset lifecycle and governance

    Treat a training dataset as a versioned software artifact. Record immutable source IDs,
    acquisition dates, licenses/terms, consent and privacy constraints, filtering code,
    deduplication method, language/domain labels, and cryptographic fingerprints. Keep raw,
    cleaned, formatted, tokenized, and split stages distinct so a bug can be traced rather than
    silently baked into a final Arrow file.

    Exact hash deduplication catches copies after normalization; MinHash or embedding methods
    detect near-duplicates. Deduplicate before splitting, preferably at document/source family
    level. Random row splits leak templated variants and adjacent chunks. For future-facing
    applications, time-based evaluation is more honest. For users or organizations, group splits
    test generalization and prevent identity leakage. Search training data for benchmark prompts
    and reference answers, not merely dataset names.

    Filtering is modeling: removing profanity, code, minority dialects, short responses, or
    refusals changes behavior. Document intended and unintended distribution changes. Manually
    inspect stratified samples before and after every major filter.
    """),
    ("code", r'''
    # Small deterministic dataset audit utilities.
    import hashlib, re
    def normalize(text):
        return re.sub(r"\s+", " ", text.strip().lower())
    def fingerprint(text):
        return hashlib.sha256(normalize(text).encode()).hexdigest()[:12]

    audit_rows = [
        {"source": "a", "text": "Gradient accumulation uses microbatches."},
        {"source": "b", "text": " gradient   accumulation uses microbatches. "},
        {"source": "c", "text": "Activation checkpointing recomputes activations."},
    ]
    seen = {}
    for row in audit_rows:
        key = fingerprint(row["text"])
        print(row["source"], key, "duplicate_of", seen.get(key))
        seen.setdefault(key, row["source"])
    '''),
    ("md", r"""
    ## 7.4 Chat templates and assistant-only loss

    A conversational dataset is structured records, not preformatted strings. Preserve roles
    and content, then render with the target tokenizer's chat template. Templates determine BOS,
    role delimiters, EOS placement, generation prompts, and sometimes tool syntax. Training with
    one template and serving with another is distribution shift. Confirm a round trip on multi-
    turn, system, tool, empty, and long examples.

    Assistant-only loss requires knowing which rendered tokens belong to assistant messages.
    Searching decoded text for a delimiter is fragile because delimiters can appear in content
    and token boundaries differ. Prefer templates that return assistant masks or a data collator
    designed for the exact template. Decide whether assistant headers and EOS are targets. Tool
    calls and reasoning fields may require distinct policy. Print tokens alongside masks for a
    handful of examples before training.
    """),
    ("code", r'''
    # Inspect every rendered token; adapt the target column to the template's mask support.
    sample = rows[0]["messages"]
    rendered_ids = tokenizer.apply_chat_template(sample, tokenize=True,
                                                  add_generation_prompt=False)
    print("idx | id | token")
    for i, token_id in enumerate(rendered_ids):
        piece = tokenizer.decode([token_id]).replace("\n", "\\n")
        print(f"{i:3d} | {token_id:6d} | {piece!r}")
    '''),
    ("md", r"""
    ## 7.5 Length policy, packing, and throughput

    Plot token lengths after final rendering. Choose maximum length using percentile coverage,
    task requirements, available memory, and truncation semantics. “Keep end” may preserve the
    answer but lose the question; “keep start” can delete the answer. Filter or construct windows
    deliberately when neither is safe. For long documents, sample spans or pack naturally rather
    than always taking prefixes.

    Packing raises utilization but changes boundaries. Standard causal packing allows later
    examples to attend to earlier ones, even if loss is separated by EOS. Block-diagonal
    attention prevents contamination but needs compatible kernels/collators. Position IDs may
    reset or continue. Sequence packing also changes batch-length variance and tokens/update.
    Report effective *tokens* per optimizer step, not only examples. Dynamic padding plus length
    bucketing is a simpler intermediate optimization.

    **Preflight reference:** schema validation; role alternation; nonempty assistant targets;
    special-token correctness; length/truncation report; duplicate/leakage report; masked-token
    percentage; packed utilization; and decoded random samples from the actual dataloader.
    """),
)

lesson(9,
    ("md", r"""
    ## 9.3 Reading a pretraining run like an experiment

    The unit of progress is tokens, not epochs. An epoch over a duplicated toy corpus is merely a
    convenient bounded loop; large pretraining corpora may be traversed once or not even have a
    meaningful epoch boundary. Record unique documents, raw and post-filter tokens, repeated tokens,
    sequence length, padding fraction, optimizer updates, effective token batch, and total compute.
    Two “one epoch” runs can represent radically different amounts of learning.

    Cross-entropy averages surprise at the target tokens. Perplexity is its exponential and is only
    comparable when tokenizer, tokenization, evaluation text, boundary handling, and masking match.
    Byte-level perplexity and GPT-style subword perplexity are therefore not directly comparable.
    Generation samples are useful qualitative probes, but a lucky completion cannot replace held-out
    loss and capability tests. Before trusting a curve, deliberately overfit one batch, verify the
    input/label shift, inspect masks, and compare a checkpoint round trip on fixed logits.

    The tiny corpus repeats phrases so a learner can see a result quickly. This creates memorization
    and distribution leakage by design. A real split occurs before deduplication-aware packing and is
    separated by document, source, author, or time where appropriate. Data cards should capture
    provenance, license, languages, filtering, personal-information policy, and known blind spots.
    """),
    ("code", r'''
    # Translate loop settings into the quantities a run report should state.
    batch_size = train_loader.batch_size
    updates = len(train_loader)
    tokens_per_update = batch_size * SEQ_LEN
    print({"optimizer_updates": updates,
           "nominal_tokens_per_update": tokens_per_update,
           "nominal_epoch_tokens": updates * tokens_per_update,
           "unique_source_documents": len(train_docs)})
    ''') ,
    ("md", r"""
    ## 9.4 Scaling beyond the demonstration

    Scaling first stresses data delivery and failure recovery. Shard immutable tokenized data; shuffle
    reproducibly across workers; save model, optimizer, scheduler, scaler, RNG, and sampler position;
    and test resumption early. Mixed precision reduces memory and raises throughput on supported GPUs.
    Gradient accumulation raises effective batch without fitting more activations at once. Activation
    checkpointing trades recomputation for memory. DDP replicates the model, while FSDP/ZeRO shard
    state; tensor and pipeline parallelism become relevant when layers cannot fit on one accelerator.

    Monitor validation loss by domain, gradient norm, learning rate, tokens/second, hardware
    utilization, data-loader stalls, numerical overflows, memory, and checkpoint health. Scaling a
    silent label bug only makes an expensive bug. Notebooks 10 and 11 develop optimization and
    distributed mechanics after this end-to-end anchor.

    Initialization and optimization interact with scale. Residual branches, normalization, embedding
    variance, learning-rate warmup, and weight decay determine whether signals remain numerically useful
    through depth. Seed every relevant generator for debugging, but repeat important conclusions across
    seeds because a single tiny run has high variance. Inspect parameter update norms relative to
    parameter norms, not only scalar loss. If loss falls implausibly quickly, check duplicated validation
    text and boundary leakage. If it remains near the uniform baseline `log(vocabulary_size)`, verify
    labels, causal alignment, and optimizer updates before changing the architecture.
    """),
)

lesson(10,
    ("md", r"""
    ## 10.3 Hugging Face object boundaries

    A tokenizer maps text and special-token conventions to IDs. A configuration specifies architecture
    and dimensions. A model class implements computation and owns parameters. A checkpoint supplies
    parameter values. `AutoModelForCausalLM.from_config(config)` selects the architecture and initializes
    new weights; `from_pretrained(path_or_id)` resolves configuration and weights from a directory or
    Hub repository. Confusing these calls can accidentally turn a from-scratch experiment into continued
    pretraining—or discard expensive weights.

    A language-model collator normally copies input IDs into labels and marks padding as ignored. Because
    GPT-2 reuses EOS as PAD and our equal-length blocks need no padding, the explicit collator deliberately
    retains genuine EOS labels. Blindly masking by token ID would erase every document boundary. Model
    forward shifts logits and labels internally when `labels` are supplied. Inspect this behavior for any
    custom architecture rather than shifting twice. `save_pretrained` writes a portable artifact, but optimizer,
    scheduler, data position, RNG state, metrics, and code revision still need a training checkpoint or
    experiment record if exact resumption matters.
    """),
    ("code", r'''
    sample = next(iter(train_loader))
    sample = {key: value.to(device) for key, value in sample.items()}
    with torch.no_grad(): out = model(**sample)
    print("logits:", tuple(out.logits.shape), "labels:", tuple(sample["labels"].shape))
    print("ignored label positions:", int((sample["labels"] == -100).sum()))
    print("config model_type:", model.config.model_type,
          "tied embeddings:", model.config.tie_word_embeddings)
    ''') ,
    ("md", r"""
    ## 10.4 From a local artifact to a governed model release

    A useful repository should include a model card, exact base/tokenizer references, licenses, dataset
    lineage, intended and excluded uses, training hyperparameters, evaluation tables, limitations, and
    example loading code. Pin revisions when reproducing a run. Safe tensor serialization avoids pickle
    execution for weight files, but consumers must still review custom code, dependencies, and model
    provenance. A Hub token is needed only for gated/private resources or upload; it must come from
    Colab Secrets or environment variables and must never be embedded in the notebook.

    Continued pretraining starts from learned weights and exposes them to more raw domain text. It can
    improve domain likelihood yet cause catastrophic forgetting or alter safety behavior. Instruction
    tuning instead trains desired prompt-response behavior, usually masking prompt labels. The next
    lesson maps those branches before later notebooks implement LoRA SFT and DPO.

    Initialization is part of the reproducibility contract. A fixed PyTorch seed makes this lesson easier
    to debug, but accelerator kernels and software versions can still alter exact results. Store package
    versions, configuration JSON, corpus fingerprint, seed, precision, and hardware alongside metrics.
    When comparing architectures, match useful token or compute budgets and repeat multiple seeds; one
    lucky small run is weak evidence.

    Generation before and after training is intentionally sampled. For a strict regression test, compare
    teacher-forced loss or fixed logits on frozen inputs. For a model card, show multiple representative
    prompts, disclose decoding parameters, and include failures. Never present a memorized training phrase
    as evidence of broad instruction following. This base artifact has not learned a chat template,
    alignment policy, factual coverage, calibrated uncertainty, or reliable stopping behavior.
    """),
)

lesson(11,
    ("md", r"""
    ## 11.6 A capability stack rather than one magic checkpoint

    Separate knowledge in weights from knowledge supplied in context, behaviors learned during tuning,
    and guarantees enforced by software. Continued pretraining may internalize stable domain patterns;
    retrieval supplies changing evidence; SFT teaches response conventions; preference optimization
    shifts ambiguous quality judgments; tools execute exact or current operations; constrained decoding
    enforces syntax; and the application control plane owns identity, permissions, budgets, and audit.
    Asking training to solve a systems problem creates brittle assurances.

    The stages also have distinct failure modes. Domain adaptation can forget general skills. SFT can
    overfit a style and reduce diversity. Preference training can exploit judge shortcuts or make answers
    verbose and overconfident. Verifier-based RL can reward-hack an incomplete checker. Distillation can
    transfer teacher errors. Quantization can produce capability regressions on sensitive tasks. Tool
    training can yield syntactically valid but unauthorized calls. Maintain stage-specific diagnostics
    alongside end-to-end product evaluation.
    """),
    ("code", r'''
    release_gates = {
        "continued pretraining": ["domain perplexity", "general retention", "memorization/privacy"],
        "SFT": ["instruction adherence", "format validity", "general retention"],
        "preference/RL": ["blinded preference", "reward hacking", "calibration", "safety"],
        "reasoning": ["pass@1", "pass@N", "verifier robustness", "tokens and latency"],
        "tools": ["call accuracy", "argument validity", "permission denial", "recovery"],
        "deployment": ["quality parity", "throughput", "tail latency", "resource saturation"],
    }
    for stage, gates in release_gates.items(): print(stage, "->", ", ".join(gates))
    ''') ,
    ("md", r"""
    ## 11.7 Choosing an optimization family

    Use SFT when you have trusted target demonstrations. Use pairwise preference methods when producing
    a single ideal answer is hard but comparison is reliable. Use reward modeling and online RL when the
    policy must explore beyond a static set and the reward can be audited. Use verifiable RL where a
    deterministic or independently trusted checker captures the real objective. Do not use RL merely
    because a task sounds difficult: unstable rewards and weak evaluation can make it an expensive route
    to worse behavior.

    Reasoning data deserves special scrutiny. Correct final answers can accompany invalid intermediate
    steps, and plausible traces can accompany wrong answers. Outcome supervision is scalable where final
    answers are checkable; process supervision offers denser feedback but is costly and can encode one
    preferred solution style. Synthetic traces multiply coverage but also correlated teacher errors.
    Filter with independent checks, retain diverse strategies, and test on uncontaminated problems.

    Finally, optimize the inference policy jointly with the model. Temperature, maximum thinking budget,
    early stopping, number of candidates, tool limits, and selection method determine quality and cost.
    Report accuracy against generated tokens and wall-clock latency—not accuracy alone—so a “better”
    reasoning configuration does not conceal a tenfold serving bill.
    """),
)

lesson(12,
    ("md", r"""
    ## 12.3 AdamW under the hood

    Adam tracks exponentially decayed first and second gradient moments:
    \(m_t=\beta_1m_{t-1}+(1-\beta_1)g_t\) and
    \(v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2\). Bias correction compensates for zero
    initialization; the update divides by \(\sqrt{\hat v_t}+\epsilon\). AdamW applies weight
    decay separately from that adaptive gradient update. Biases and normalization scale
    parameters are often excluded from decay.

    The learning rate is the most consequential hyperparameter, but its safe range depends on
    model scale, batch/tokens per update, optimizer, precision, data, adapter/full tuning, and
    initialization. Warmup prevents large early adaptive updates while moments are unreliable.
    Linear or cosine decay is common; constant schedules can work for short fine-tuning. Specify
    warmup in steps or token proportion and log the actual schedule.
    """),
    ("code", r'''
    # Visualize common warmup/decay schedules independent of Trainer.
    import math, matplotlib.pyplot as plt
    total, warmup = 1000, 100
    def multiplier(step, kind):
        if step < warmup: return step / max(warmup, 1)
        progress = (step - warmup) / (total - warmup)
        if kind == "linear": return 1 - progress
        if kind == "cosine": return .5 * (1 + math.cos(math.pi * progress))
        return 1.0
    steps = range(total)
    for kind in ["constant", "linear", "cosine"]:
        plt.plot(steps, [multiplier(s, kind) for s in steps], label=kind)
    plt.xlabel("optimizer step"); plt.ylabel("LR multiplier"); plt.legend(); plt.show()
    '''),
    ("md", r"""
    ## 12.4 A production-quality training loop

    Set model mode deliberately: `train()` enables dropout; `eval()` disables it. Move batches
    to the correct device, use autocast for selected precision, divide loss for accumulation,
    backward, unscale before gradient clipping, update only at accumulation boundaries, advance
    the scheduler per optimizer update, and zero gradients with `set_to_none=True`. Handle a
    final partial accumulation window correctly. Evaluation runs under inference/no-grad mode
    and restores training mode afterward.

    Distributed training adds sampler epochs, synchronized metrics, and main-process-only writes.
    Reproducibility needs Python/NumPy/framework seeds, data sampler state, deterministic choices
    where available, and environment capture. Exact bitwise reproducibility may still fail across
    hardware and kernels; define the level you require. Checkpoints need model/adapters, optimizer,
    scheduler, scaler, RNG, step/epoch, data position, and config. Test resume, do not merely save.
    """),
    ("code", r'''
    # A reusable evaluation function illustrating reduction by examples.
    @torch.inference_mode()
    def evaluate_classifier(model, batches):
        was_training = model.training
        model.eval()
        total_loss = total_correct = total_items = 0
        for x, target in batches:
            logits = model(x)
            total_loss += torch.nn.functional.cross_entropy(
                logits, target, reduction="sum").item()
            total_correct += (logits.argmax(-1) == target).sum().item()
            total_items += target.numel()
        model.train(was_training)
        return {"loss": total_loss / total_items,
                "accuracy": total_correct / total_items}

    validation = [(torch.randn(8, 16), torch.randint(0, 8, (8,))) for _ in range(3)]
    print(evaluate_classifier(model, validation))
    '''),
    ("md", r"""
    ## 12.5 Reading training curves and running ablations

    Compare train and validation loss on identical reduction/token policies. A sudden loss spike
    may be a rare long batch, bad record, overflow, schedule discontinuity, or distributed issue;
    log sample IDs and tokens/update. Smooth curves for visualization but retain raw values.
    Validation loss can improve while task behavior regresses, so run task evals at checkpoints.
    Inspect qualitative generations under fixed decoding.

    Change one factor per ablation when possible: LR, warmup, effective batch, max length, masking,
    LoRA rank/targets, data mixture. Report compute/tokens, not only epochs. Compare against a
    no-training baseline and a simple prompting/RAG baseline. Stop based on held-out evidence and
    budget, not because a predetermined epoch count completed. Archive failed runs: knowing that
    a configuration diverged is useful evidence when metadata is complete.

    **Minimum run record:** code/data/model revisions, tokenizer/template, seed, precision,
    hardware/world size, optimizer/schedule, effective tokens/update, step/tokens seen, gradient
    statistics, checkpoint IDs, eval results, and wall-clock/cost.
    """),
)

lesson(13,
    ("md", r"""
    ## 13.4 A more complete memory model

    Parameter memory includes weights, gradients, optimizer moments, and sometimes FP32 master
    weights. Activation memory scales with batch, sequence length, width, layers, attention
    implementation, and which tensors backward saves. Temporary buffers, allocator fragmentation,
    communication buckets, kernels, and the CUDA context also matter. Peak memory—not steady-state
    snapshots—determines whether a step succeeds.

    Estimate analytically, then measure `max_memory_allocated` and `max_memory_reserved` after
    resetting peak statistics. Reserved memory is the allocator pool and can exceed active tensor
    memory. An OOM after several steps may come from variable length, evaluation generation,
    leaked graph references, logging tensors without `.item()`, or checkpoint/save spikes.
    Sequence length often has nonlinear impact because naive attention intermediates are quadratic.
    FlashAttention changes that intermediate-memory term, not all activations.
    """),
    ("code", r'''
    # Compare rough parameter-state scenarios.
    def state_memory(params, weight=2, grad=2, master=0, moments=8):
        return params * (weight + grad + master + moments)
    params = 7_000_000_000
    scenarios = {
        "mixed precision AdamW": state_memory(params, 2, 2, 4, 8),
        "bf16 weights+grads, fp32 moments": state_memory(params, 2, 2, 0, 8),
        "weights only inference": state_memory(params, 2, 0, 0, 0),
    }
    for name, value in scenarios.items(): print(name, f"{value/2**30:.1f} GiB")
    print("These exclude activations, buffers, fragmentation, and communication.")
    '''),
    ("md", r"""
    ## 13.5 Precision and checkpointing mechanics

    FP16 has limited exponent range; gradient scaling multiplies loss before backward, detects
    overflow, skips invalid updates, and adjusts scale. BF16 retains FP32's exponent width with
    fewer mantissa bits and generally avoids loss scaling, but hardware support matters. FP32
    accumulation may still be used inside reductions. TF32 changes eligible NVIDIA matrix
    operations while tensors remain FP32. Precision is per-operation policy, not one global dtype.

    Activation checkpointing partitions the graph and saves only boundary inputs, replaying
    forward operations during backward. It must preserve RNG behavior for dropout and avoid
    side effects. More segments save more memory but add recompute and overhead. “Gradient
    checkpointing” is unrelated to saving training checkpoints. Combine it with FlashAttention,
    accumulation, and length/batch changes only after measuring interactions.
    """),
    ("code", r'''
    # Demonstrate checkpointed forward/backward on CPU or GPU.
    from torch.utils.checkpoint import checkpoint
    device = "cuda" if torch.cuda.is_available() else "cpu"
    deep = torch.nn.Sequential(*[
        torch.nn.Sequential(torch.nn.Linear(512, 512), torch.nn.GELU())
        for _ in range(8)]).to(device)
    x = torch.randn(16, 512, device=device, requires_grad=True)
    def run_segment(start, end, value):
        for layer in list(deep.children())[start:end]: value = layer(value)
        return value
    h = checkpoint(lambda z: run_segment(0, 4, z), x, use_reentrant=False)
    y = checkpoint(lambda z: run_segment(4, 8, z), h, use_reentrant=False)
    y.square().mean().backward()
    print("output", y.shape, "input grad norm", x.grad.norm().item())
    '''),
    ("md", r"""
    ## 13.6 Distributed strategy decision guide

    Use DDP when one model replica plus optimizer fits per GPU and throughput scales with more
    data. FSDP/ZeRO shard states when replicas do not fit; sharding stages trade memory for
    communication and more complex checkpointing. Tensor parallelism splits layer matrices and
    benefits from fast intra-node interconnect. Pipeline parallelism splits layer ranges but has
    bubble and microbatch scheduling costs. Sequence/context parallelism divides sequence-related
    work. Expert parallelism distributes MoE experts. Large jobs combine dimensions.

    Network topology is part of the algorithm. Keep high-frequency tensor-parallel collectives on
    fast links; use data parallel across slower nodes where possible. Measure scaling efficiency
    against one device, communication time, idle/bubble time, tokens/sec, convergence per token,
    and checkpoint duration. A configuration that processes tokens faster but changes effective
    batch or numerical behavior is not a controlled speed comparison.

    In Colab, focus on single-GPU accumulation, checkpointing, mixed precision, and adapters.
    Multi-GPU examples belong on controlled Linux infrastructure with Accelerate configuration
    committed alongside the run.
    """),
)

lesson(14,
    ("md", r"""
    ## 14.4 LoRA mechanics and target selection

    For a frozen projection \(y=xW^T\), LoRA adds
    \((\alpha/r)xA^TB^T\). One factor is commonly initialized randomly and the other to zero,
    so the adapter initially leaves model behavior unchanged. Rank controls capacity; alpha
    scales the update; dropout regularizes adapter input. Rank and alpha are coupled through the
    scaling convention. Adapter parameters are small, but forward/backward through the frozen
    base and activations still consume compute/memory.

    Attention Q/V targets are a small baseline. Adding K/O and MLP projections increases
    capacity and trainable parameters. Module names vary by architecture; print matched modules
    and trainable counts rather than trusting a copied pattern. Rank-stabilized LoRA, DoRA, and
    adaptive-rank methods alter parameterization, but data and evaluation usually matter more
    than chasing variants prematurely.
    """),
    ("code", r'''
    # Inspect candidate projection names and calculate adapter coverage without training.
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    hidden = cfg.hidden_size
    intermediate = cfg.intermediate_size
    rank = 16
    matrices = {
        "q_proj": (hidden, hidden), "k_proj": (hidden, hidden),
        "v_proj": (hidden, hidden), "o_proj": (hidden, hidden),
        "gate_proj": (hidden, intermediate), "up_proj": (hidden, intermediate),
        "down_proj": (intermediate, hidden),
    }
    for name, (inp, out) in matrices.items():
        print(name, f"{rank*(inp+out):,} LoRA params/layer")
    '''),
    ("md", r"""
    ## 14.5 QLoRA in detail

    QLoRA quantizes the *frozen* base weights, commonly to 4-bit NormalFloat (NF4), while LoRA
    adapters train in BF16/FP16. Double quantization compresses quantization constants. Paged
    optimizers can handle memory spikes. Dequantization occurs for computation; this is not
    equivalent to training 4-bit adapter values. Compute dtype, quantization type, device map,
    and hardware kernel support must be explicit.

    Quantization reduces base weight memory, but activations, adapter gradients, and optimizer
    states remain. Quality can regress for some models/tasks, and merging adapters generally
    requires a suitable higher-precision base. `bitsandbytes` is CUDA-oriented; platform support
    changes. In Colab, verify the installed library, GPU compute capability, loaded parameter
    dtypes, and actual allocated memory. A config flag without inspecting the loaded model is not
    proof that QLoRA is active.
    """),
    ("code", r'''
    # Reference QLoRA configuration (construction only; use inside the guarded GPU cell).
    try:
        from transformers import BitsAndBytesConfig
        import torch
        qlora_quantization = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
        print(qlora_quantization)
    except Exception as exc:
        print("QLoRA configuration unavailable in this runtime:", type(exc).__name__)
    '''),
    ("md", r"""
    ## 14.6 SFT experiment design and deployment

    Establish base-model results using the exact inference template. Train on behavior the
    prompt/RAG baseline cannot achieve reliably. Use validation sources disjoint from training,
    plus a frozen capability/safety suite. Tune LR, effective tokens/update, epochs/tokens, rank,
    targets, max length, masking, and data mixture conservatively. Small curated sets can overfit
    quickly; more epochs are not automatically useful.

    At inference, load base plus adapter dynamically or merge the update. Dynamic adapters enable
    multiple tasks but add routing/operational complexity. Merging simplifies serving but produces
    a new weight artifact and can interact with quantization. Record base commit, adapter config,
    tokenizer/template, library versions, data fingerprint, and eval report. Test that a fresh
    process can reconstruct outputs from published artifacts.

    **Failure diagnosis:** no behavior change—mask/targets/LR/modules; memorization—duplicates or
    too many epochs; broad regression—data narrowness/LR/targets; malformed chat—template/EOS;
    OOM—length/activations/precision; slow training—padding, checkpoint recompute, dataloader, or
    unsupported kernels.
    """),
)

lesson(15,
    ("md", r"""
    ## 15.3 Preference-data construction

    A preference record needs the same prompt/context with chosen and rejected responses. If
    prompts differ, the comparison is confounded. Rejections should be plausible alternatives;
    trivial bad answers teach superficial separation. Capture criterion, rater, confidence/tie,
    response order, model sources, and timestamps. Randomize display order and measure agreement.
    Resolve whether “preferred” means more correct, safer, more concise, or simply stylistically
    liked—these objectives can conflict.

    Audit shortcuts before training: chosen length, headings, disclaimers, refusal phrases,
    citations, model-specific style, and lexical markers. A policy can optimize these without
    improving substance. Split by prompt/source, deduplicate responses, and keep a human-reviewed
    test set. Synthetic AI preferences scale cheaply but inherit judge biases; mix them with human
    checks and task-verifiable signals where possible.
    """),
    ("code", r'''
    # Preference shortcut audit on a toy set.
    pairs = [
        {"chosen": "The answer is 4.", "rejected": "I think it might perhaps be five."},
        {"chosen": "Insufficient evidence.", "rejected": "Definitely Paris, with complete certainty."},
        {"chosen": "Use two microbatches.", "rejected": "Use one batch."},
    ]
    for i, pair in enumerate(pairs):
        c, r = pair["chosen"], pair["rejected"]
        print(i, {"chosen_chars": len(c), "rejected_chars": len(r),
                  "chosen_words": len(c.split()), "rejected_words": len(r.split())})
    '''),
    ("md", r"""
    ## 15.4 Understanding beta, margins, and reference behavior

    DPO compares the policy's chosen/rejected log-probability margin with the reference margin.
    If the policy already favors the chosen response more strongly than the reference, its DPO
    logit is positive and loss falls. Beta controls sensitivity/regularization conventionally:
    larger beta makes a given margin difference produce a more saturated classification signal.
    Monitor chosen/rejected rewards, margins, accuracy, log probabilities, and KL-like drift—not
    only scalar loss.

    The reference policy anchors behavior. With PEFT, implementations may use the base model or
    disabled adapter as reference to avoid a full duplicate, but verify library semantics and
    memory. DPO assumes preference data follows a Bradley–Terry-style logistic model and does not
    explicitly optimize absolute answer quality. Both candidates can be poor; adding SFT/NLL
    components or quality filtering can help. Preference optimization can also reduce diversity
    or exploit length, so downstream evals remain decisive.
    """),
    ("code", r'''
    # Visualize DPO loss across policy-minus-reference preference margins.
    import torch, matplotlib.pyplot as plt
    margins = torch.linspace(-5, 5, 201)
    for beta in [0.05, 0.1, 0.5, 1.0]:
        losses = -torch.nn.functional.logsigmoid(beta * margins)
        plt.plot(margins, losses, label=f"beta={beta}")
    plt.xlabel("policy preference margin - reference margin")
    plt.ylabel("DPO pair loss"); plt.legend(); plt.show()
    '''),
    ("md", r"""
    ## 15.5 Preference optimization family and evaluation

    PPO/RLHF trains a reward model and optimizes policy actions with an online RL algorithm;
    it is flexible but operationally complex. DPO gives a direct offline objective. IPO changes
    the loss to address overfitting behavior. KTO can learn from desirable/undesirable examples
    without pairs. ORPO/SimPO and other variants change reference or SFT coupling. GRPO-style
    methods compare groups and are used with verifiable/reward signals. Names evolve quickly;
    compare assumptions, data requirements, stability, compute, and empirical evals.

    Evaluate pairwise win rate with randomized order, objective task metrics, calibration of
    refusals, verbosity/length, safety slices, general capability retention, and generation
    diversity. Use multiple decoding settings because preference training changes distribution
    shape. Compare SFT checkpoint, preference checkpoint, and base. Inspect regressions rather
    than reporting a single judge win rate. Roll out gradually: preference optimization can
    strongly change tone and refusal behavior even when average benchmarks improve.
    """),
)

lesson(17,
    ("md", r"""
    ## 17.3 Repository anatomy and reproducibility

    `config.json` describes architecture, but model-specific fields still require compatible
    Transformers code. Tokenizer artifacts may include `tokenizer.json`, vocabulary/merges or a
    SentencePiece model, special-token mappings, and `tokenizer_config.json` containing a chat
    template. Generation defaults can live in `generation_config.json`. Weights are commonly
    sharded Safetensors files plus an index. Processor files configure image/audio preprocessing.

    A model ID points to a mutable repository branch unless `revision` is pinned to a commit.
    Pin model, tokenizer, dataset, and code revisions; store the resolved commit. Safetensors
    avoids arbitrary pickle execution for weights, but `trust_remote_code=True` imports Python
    from the repository. Review/pin that code and run it with appropriate isolation. Model cards
    should state training data, intended use, limitations, license, metrics, and environmental or
    ethical considerations, but completeness varies. Absence of a warning is not evidence of
    suitability.
    """),
    ("code", r'''
    # List files and classify the artifact surface without downloading weights.
    from huggingface_hub import list_repo_files
    files = list_repo_files(MODEL_ID, repo_type="model", revision=info.sha)
    groups = {"weights": [], "tokenizer": [], "config": [], "code": [], "other": []}
    for name in files:
        lower = name.lower()
        if lower.endswith((".safetensors", ".bin", ".gguf")): group = "weights"
        elif any(x in lower for x in ["tokenizer", "vocab", "merges", "sentencepiece"]): group = "tokenizer"
        elif lower.endswith((".json", ".yaml", ".yml")): group = "config"
        elif lower.endswith(".py"): group = "code"
        else: group = "other"
        groups[group].append(name)
    for group, names in groups.items(): print(group, names[:8], "..." if len(names) > 8 else "")
    '''),
    ("md", r"""
    ## 17.4 Loading, devices, dtypes, and memory

    `from_pretrained` resolves config, downloads/caches files, constructs modules, and loads
    tensors. `dtype="auto"` follows checkpoint/config behavior; inspect actual parameter dtypes.
    `.to(device)` places the complete model on one device. `device_map="auto"` uses Accelerate
    to place layers across devices/CPU, which is useful for fitting but can be slow and is not a
    training strategy. Quantization configurations alter storage and kernels. `low_cpu_mem_usage`
    and sharded loading reduce temporary host memory.

    `model.eval()` disables dropout but does not disable gradients; use `torch.inference_mode()`.
    Input tensors must share a compatible device with the first model layers. Decode only newly
    generated token IDs, not the full prompt, when presenting output. Inspect finish condition and
    output length. Cache location and revision pinning matter in ephemeral Colab environments;
    gated models require license acceptance and token permissions before download.
    """),
    ("code", r'''
    # Inspect the loaded model rather than trusting requested settings.
    first = next(model.parameters())
    parameter_count = sum(p.numel() for p in model.parameters())
    bytes_used = sum(p.numel() * p.element_size() for p in model.parameters())
    print("class:", model.__class__.__name__)
    print("parameters:", f"{parameter_count:,}")
    print("parameter storage:", f"{bytes_used/2**20:.1f} MiB")
    print("first parameter dtype/device:", first.dtype, first.device)
    print("context configured:", getattr(model.config, "max_position_embeddings", None))
    print("attention implementation:", getattr(model.config, "_attn_implementation", None))
    '''),
    ("md", r"""
    ## 17.5 Local, Inference Providers, Endpoints, and dedicated serving

    Local Transformers is transparent and ideal for learning or batch experiments, but it does
    not provide multi-user scheduling. Inference Providers route a common client call to hosted
    providers; supported models/features and billing vary. Dedicated Hugging Face Inference
    Endpoints provide managed replicas for chosen models. TGI and vLLM are specialized servers.
    Choose based on model support, privacy boundary, hardware control, concurrency, latency/SLA,
    observability, autoscaling, and sustained utilization—not “open versus closed” alone.

    Build an application adapter that owns model ID, revision, timeout, retry policy, decoding,
    schema validation, and normalized usage/error records. Remote failures include authentication,
    gating, no compatible provider, cold starts, quota/rate limits, transient 5xx, unsupported
    structured output/tools, and context overflow. Catch specific error classes where possible and
    preserve a safe diagnostic. Never silently switch models in a quality-sensitive workflow.

    **Model selection reference:** validate license; inspect tokenizer/template; establish memory;
    run task evals; benchmark prompt/output distributions; test safety/languages; pin revision;
    and record operational compatibility before promotion.
    """),
)

lesson(18,
    ("md", r"""
    ## 18.4 Sampling algorithms step by step

    Repetition penalties modify logits before filtering. Temperature rescales. Top-k masks all but
    k highest logits. Top-p sorts probabilities and retains the smallest prefix whose cumulative
    mass crosses p. Min-p keeps tokens relative to the best probability. Typical sampling favors
    tokens near expected information content. The order and exact implementation are library
    behavior; inspect `GenerationConfig` for the installed version.

    Greedy decoding is not the same as globally most likely sequence: it selects the best token at
    each step. Beam search tracks multiple sequence hypotheses and is useful for constrained
    sequence tasks, but can produce generic text for open-ended chat. Sampling gives a distribution
    of outcomes. For evaluation, fix seeds and repeat stochastic calls; for production, choose
    parameters based on measured task success, diversity, safety, latency, and output length.
    """),
    ("code", r'''
    # Implement top-k/top-p filtering on one logit vector.
    def filter_logits(logits, top_k=None, top_p=None):
        values = logits.clone()
        if top_k:
            threshold = torch.topk(values, min(top_k, values.numel())).values[-1]
            values[values < threshold] = -torch.inf
        if top_p is not None:
            sorted_logits, indices = torch.sort(values, descending=True)
            probs = sorted_logits.softmax(-1)
            remove = probs.cumsum(-1) - probs > top_p
            values[indices[remove]] = -torch.inf
        return values

    demo = torch.tensor([4., 3., 2., 1., 0.])
    for kwargs in [{}, {"top_k": 2}, {"top_p": .8}, {"top_k": 4, "top_p": .8}]:
        filtered = filter_logits(demo, **kwargs)
        print(kwargs, filtered.softmax(-1))
    '''),
    ("md", r"""
    ## 18.5 Stopping, length, and reproducibility

    `max_new_tokens` caps generated tokens and is generally clearer than total `max_length`.
    EOS lets the model stop naturally; custom stop strings require token-aware or incremental
    matching and may span token boundaries. A finish reason of length means the result may be
    incomplete. Forced minimum lengths can encourage filler. For structured output, reserve enough
    tokens to close the structure and still validate it.

    Seeds control random number streams but full determinism also depends on hardware, kernels,
    batching, precision, library versions, and server behavior. Dynamic batching can alter numeric
    paths. Store raw outputs and complete generation config. For regression tests, greedy decoding
    is convenient but may not represent the production sampling distribution; maintain both a
    deterministic suite and repeated stochastic quality estimates.
    """),
    ("code", r'''
    # Compare deterministic and sampled continuations using the already loaded small model.
    prompt = tok("A reliable evaluation should", return_tensors="pt").to(lm.device)
    configs = [
        {"do_sample": False},
        {"do_sample": True, "temperature": .7, "top_p": .9},
        {"do_sample": True, "temperature": 1.2, "top_k": 20},
    ]
    for i, config in enumerate(configs):
        torch.manual_seed(123)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(123)
        result = lm.generate(**prompt, max_new_tokens=30, **config)
        print(i, tok.decode(result[0, prompt["input_ids"].shape[1]:], skip_special_tokens=True))
    '''),
    ("md", r"""
    ## 18.6 Batching and streaming implementation reference

    Throughput batching groups prompts into one forward pass. Padding waste grows with length
    variance, so bucket by length. Batch size is constrained by prompt tokens, expected output,
    KV cache, and allocator workspace—not examples alone. One long generation keeps the batch
    active unless the engine supports continuous batching. Local `generate()` is not a substitute
    for a concurrent inference server.

    Streaming transports deltas, not independent complete messages. Accumulate by choice/index;
    handle role/tool deltas, empty chunks, usage/final metadata, finish reasons, disconnects, and
    cancellation. Rendering untrusted HTML/Markdown incrementally introduces security concerns.
    Measure queue time and TTFT separately from generation time. A canceled client should cancel
    upstream compute where supported rather than merely stop displaying tokens.

    Structured generation constrains syntax according to JSON Schema/grammar, but semantic
    validation remains application work. Parse once complete, validate types/ranges/enums and
    business rules, reject extra properties, and define retries. Record provider fallback from
    strict schema to JSON mode because it changes reliability.
    """),
)

lesson(19,
    ("md", r"""
    ## 19.3 How embedding models are trained

    Bi-encoders learn geometry from positive and negative pairs using contrastive losses. In-batch
    negatives make other examples' documents serve as negatives; quality depends on batch size and
    false-negative rate. Multiple Negatives Ranking Loss and InfoNCE-like objectives raise positive
    similarity relative to negatives. Hard negatives—topically related but irrelevant documents—
    teach fine distinctions, while accidental true positives mislabeled negative damage training.

    Query/document encoders may share weights or use asymmetric prompts such as `query:` and
    `passage:`. Some models require instruction prefixes. Pooling can use CLS, mean token pooling,
    or learned mechanisms. Normalization changes the score from magnitude-sensitive dot product to
    cosine. Matryoshka-trained embeddings support truncating dimensions with graceful degradation;
    arbitrary dimension truncation does not. Follow the model card's encoding recipe exactly.
    """),
    ("code", r'''
    # Inspect score distribution, margins, and retrieval confidence diagnostics.
    for qi, query in enumerate(queries):
        order = np.argsort(-scores[qi])
        top = scores[qi, order[:3]]
        print(query)
        print(" top IDs:", order[:3].tolist(), "scores:", top.round(3).tolist(),
              "top1-top2 margin:", round(float(top[0] - top[1]), 3))
    # Scores are model/index-specific; do not treat one universal threshold as confidence.
    '''),
    ("md", r"""
    ## 19.4 Lexical, dense, hybrid, and reranking systems

    BM25 rewards query-term matches using term frequency, inverse document frequency, and length
    normalization. It excels at identifiers, rare names, error codes, and exact wording. Dense
    retrieval captures paraphrase and semantic relationships but can miss rare lexical details.
    Hybrid retrieval retrieves from both, normalizes or rank-fuses results, then optionally
    reranks. Reciprocal Rank Fusion combines ranks without assuming score comparability.

    A cross-encoder jointly processes query and candidate, allowing full token interaction. It is
    too expensive for the whole corpus but effective on top 20–200 candidates. Late-interaction
    models retain token-level representations for a middle ground. The cascade's candidate count,
    deduplication, metadata filters, reranker batch size, and final k are tuned against labeled
    queries and latency. A better reranker cannot recover documents absent from candidate recall.
    """),
    ("code", r'''
    # Reciprocal-rank fusion for heterogeneous retrievers.
    def rrf(rankings, k=60):
        fused = {}
        for ranking in rankings:
            for rank, doc_id in enumerate(ranking, 1):
                fused[doc_id] = fused.get(doc_id, 0) + 1 / (k + rank)
        return sorted(fused, key=fused.get, reverse=True), fused

    dense_rank = [0, 3, 1, 2]
    lexical_rank = [3, 0, 2, 1]
    fused_order, fused_scores = rrf([dense_rank, lexical_rank])
    print(fused_order, {i: round(fused_scores[i], 4) for i in fused_order})
    '''),
    ("md", r"""
    ## 19.5 Index structures and evaluation details

    Brute-force exact search is the correctness baseline. HNSW builds a navigable graph and offers
    strong recall/latency with memory overhead. IVF partitions vectors into coarse cells and probes
    selected lists. Product quantization compresses vectors with accuracy tradeoffs. Index choice
    depends on corpus size, dimensionality, update pattern, filters, latency, memory, and target
    recall. Benchmark on production hardware with representative filters and concurrent queries.

    Recall@k asks whether any relevant document appears; precision@k measures returned relevance;
    MRR rewards the first relevant rank; nDCG handles graded relevance and multiple relevant items.
    Label incompleteness makes unjudged retrieved documents ambiguous, not automatically irrelevant.
    Slice queries by intent, lexical rarity, language, freshness, answerability, and source. Track
    embedding/index version and run reindex migrations explicitly. Mixing vectors from different
    embedding revisions silently corrupts similarity.

    **Operational checklist:** stable chunk IDs; normalized text policy; batch embedding; retry and
    checksum; dimension/normalization validation; atomic index version swap; deletion propagation;
    metadata ACL filters before exposure; and retrieval traces with sensitive content controls.
    """),
)

lesson(20,
    ("md", r"""
    ## 20.4 Ingestion and chunking as an information-retrieval problem

    Parse by source type while preserving headings, pages, tables, code blocks, timestamps, ACLs,
    canonical URL, and offsets. Remove navigation/footer duplication without erasing meaningful
    structure. Chunk boundaries should follow semantic units where possible. Fixed token windows
    are predictable; sentence/paragraph/heading chunks preserve discourse; parent-child retrieval
    embeds small child chunks but returns larger parent context. Overlap improves boundary recall
    but duplicates evidence and index/storage cost.

    Chunk size changes both retrieval and generation. Small chunks are specific but lack context;
    large chunks contain answers but embedding similarity becomes diffuse and consume prompt budget.
    Tune using supporting-span labels. Store content hashes and versioned source IDs so updates and
    deletions are traceable. Never use vector-store presence as the source of truth for authorization:
    enforce tenant/ACL metadata filters during retrieval and again before context assembly.
    """),
    ("code", r'''
    # Compare chunk sizes using stable source/offset metadata.
    for size in [8, 14, 24]:
        candidate_chunks = [c for name, text in sources.items()
                            for c in word_chunks(name, text, size=size, overlap=max(1, size//5))]
        lengths = [len(c.text.split()) for c in candidate_chunks]
        print({"size": size, "chunks": len(candidate_chunks),
               "mean_words": sum(lengths)/len(lengths),
               "duplicate_overlap_cost": sum(lengths) - sum(len(t.split()) for t in sources.values())})
    '''),
    ("md", r"""
    ## 20.5 Query transformations and context assembly

    Conversational questions may depend on prior turns; create a standalone retrieval query without
    changing intent. Multi-query retrieval generates paraphrases to improve recall but increases
    latency and false positives. HyDE embeds a hypothetical answer; decomposition retrieves for
    subquestions. These model-based transformations require evaluation because they can inject
    assumptions. Metadata filters should come from validated application state, not unrestricted
    model output.

    After retrieval/reranking, deduplicate overlapping chunks, diversify sources when appropriate,
    and allocate a token budget. Place source IDs adjacent to text. Preserve chronological or
    structural order when it matters. Separate system instructions from evidence and explicitly
    state that evidence cannot override policy. Context compression/summarization can fit more
    sources but may delete qualifiers; retain links to originals and evaluate answer support.
    """),
    ("code", r'''
    # Token-budgeted context selection with an explicit reserve.
    from transformers import AutoTokenizer
    budget_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    def select_context(results, token_budget):
        selected, used = [], 0
        for score, chunk in results:
            rendered = f"[source={chunk.source} start={chunk.start}]\n{chunk.text}"
            cost = len(budget_tokenizer.encode(rendered, add_special_tokens=False))
            if used + cost <= token_budget:
                selected.append((score, chunk)); used += cost
        return selected, used
    selected, used = select_context(results, token_budget=80)
    print("selected", len(selected), "context tokens", used)
    print(build_context(selected))
    '''),
    ("md", r"""
    ## 20.6 Grounded generation, citations, and abstention

    Ask for claim-level citations, not a decorative source list. Citation correctness has two
    dimensions: entailment (does the cited passage support the claim?) and completeness (are all
    externally verifiable claims supported?). A retrieved source may be relevant but contradictory,
    stale, or low authority. Define source precedence and surface conflicts rather than blending
    them. Models can produce valid-looking nonexistent IDs; validate citations against supplied IDs.

    Abstention is a first-class output. Train/evaluate questions with no evidence, partial evidence,
    conflicting evidence, and out-of-scope requests. Optimize selective accuracy: quality among
    answered questions versus coverage. A model's self-reported confidence is not sufficient;
    retrieval evidence, score/margin features, source coverage, and calibrated validators may inform
    routing. High-risk answers need expert or deterministic verification.
    """),
    ("code", r'''
    # Citation-ID and simple claim-support bookkeeping.
    import re
    allowed_ids = {f"{c.source}:{c.start}" for _, c in selected}
    sample_answer = "Checkpointing recomputes activations during backward [training.md:11]."
    cited = set(re.findall(r"\[([^\]]+)]", sample_answer))
    print("allowed:", allowed_ids, "cited:", cited,
          "unknown citations:", cited - allowed_ids,
          "unused evidence:", allowed_ids - cited)
    '''),
    ("md", r"""
    ## 20.7 RAG evaluation and production reference

    Build query records with answerability, reference answer/claims, supporting chunk/source IDs,
    forbidden/distractor sources, and slices. Evaluate ingestion (parse completeness), retrieval
    (recall@k/MRR/nDCG), reranking, context precision, generation correctness, faithfulness,
    citation entailment/completeness, and abstention. Run stage-oracle experiments: generate with
    gold chunks to measure generator ceiling; inspect retrieval with known supports to isolate loss.

    Production needs incremental indexing, atomic versions, freshness SLA, deletion/ACL propagation,
    embedding migrations, caches keyed by versions, traceable source snapshots, and cost/latency
    budgets. Monitor empty/low-score retrieval, answer/abstain rate, citation failures, source mix,
    latency by stage, and human feedback. Prompt injection in documents remains an application
    security issue; retrieval relevance does not imply trust.

    **Debug order:** confirm exact rendered query → filters/ACL → candidate recall → reranker →
    dedup/budget → final prompt → cited claims. Looking only at the final answer obscures the stage
    that failed.
    """),
)

lesson(22,
    ("md", r"""
    ## 22.3 Tool-schema design in depth

    Tool descriptions should state purpose, when to use/not use it, parameter semantics, units,
    defaults, allowed ranges, and result/error shape. Prefer small orthogonal tools over one giant
    “do anything” endpoint, but avoid dozens of nearly identical names. JSON Schema constrains
    syntax; application validation enforces business rules, identity, ownership, quotas, and
    cross-field invariants. Use enums and `additionalProperties: false` where supported.

    Results should be compact structured data with stable fields. Distinguish success, retryable
    error, permanent error, not-found, and authorization denied without exposing internal secrets.
    Huge raw pages consume context and carry injection risk; tools should extract bounded relevant
    fields and include provenance. Tool calls need correlation/idempotency keys for safe retries.
    Never encode hidden authority in natural-language descriptions.
    """),
    ("code", r'''
    # Validate tool arguments independently of what the model produced.
    from pydantic import BaseModel, ConfigDict, Field, ValidationError
    class CalculationArgs(BaseModel):
        model_config = ConfigDict(extra="forbid")
        expression: str = Field(min_length=1, max_length=100)

    for candidate in [{"expression": "2+2"}, {"expression": "x"*101},
                      {"expression": "2+2", "admin": True}]:
        try: print("valid", CalculationArgs.model_validate(candidate).model_dump())
        except ValidationError as exc: print("invalid", candidate, exc.errors()[0]["type"])
    '''),
    ("md", r"""
    ## 22.4 Agent loop states and control flow

    A robust loop is a state machine: receive request → call model → validate proposed calls →
    authorize → execute with timeout → normalize observation → append exact tool-call/result IDs →
    repeat or finish. Handle multiple calls, partial failure, invalid JSON, unknown tools, model
    refusal, context overflow, cancellation, and budgets. The model should see recoverable errors so
    it can revise arguments, but repeated failures terminate deterministically.

    Parallelize only independent, authorized, read-only calls. Calls whose inputs depend on prior
    results remain sequential. Side effects should use a prepare/confirm/commit pattern: the model
    drafts an action; code computes exact impact; the user approves that impact; code commits once.
    Approval cannot be an instruction buried in the same untrusted context. Store approval scope and
    expiry outside the model.
    """),
    ("code", r'''
    # A deterministic loop-policy object independent of any model provider.
    from dataclasses import dataclass, field
    @dataclass
    class AgentBudget:
        max_steps: int = 5
        max_tool_calls: int = 8
        calls: int = 0
        seen: set = field(default_factory=set)
        def admit(self, step, name, arguments):
            signature = (name, json.dumps(arguments, sort_keys=True))
            if step >= self.max_steps: return False, "step budget"
            if self.calls >= self.max_tool_calls: return False, "tool budget"
            if signature in self.seen: return False, "repeated call"
            self.seen.add(signature); self.calls += 1
            return True, "allowed"

    budget = AgentBudget()
    for step in range(3): print(step, budget.admit(step, "calculator", {"expression": "2+2"}))
    '''),
    ("md", r"""
    ## 22.5 Planning patterns and when not to use an agent

    A direct tool call is best for known workflows. A deterministic DAG/state machine is best when
    steps and transitions are known but tool results vary. An agent loop is justified when the next
    action depends on open-ended observations and flexibility outweighs added latency, cost, and
    risk. “Agent” does not require exposing hidden chain-of-thought; observable action/observation
    traces and concise decision summaries are enough for operations.

    Plan-and-execute separates a proposed plan from execution but plans become stale after new
    observations. ReAct interleaves action and observation. Reflection/reviewer loops add cost and
    can reinforce errors unless evaluated. Multi-agent systems multiply coordination and security
    surfaces; use them only when roles have genuinely separable information/capabilities.

    **Reference checklist:** bounded budgets; allowlisted tool registry; schema+semantic validation;
    external authorization; timeouts/cancellation; idempotency; bounded observations; injection
    handling; approval UX; trace/redaction; deterministic fallbacks; and trajectory-level evals.
    """),
)

lesson(24,
    ("md", r"""
    ## 24.3 Protocol lifecycle and capability negotiation

    An MCP client connects over a transport, initializes a session, exchanges protocol versions and
    capabilities, then discovers primitives. Tools are model-invocable operations. Resources are
    application-controlled readable data identified by URIs and may support templates/subscriptions.
    Prompts are reusable templates the user/host can select. Servers can request sampling from the
    host under negotiated capability and policy. Notifications communicate changes without a request.

    Transports change deployment/security. Stdio is simple and inherits the spawned process's local
    privileges/environment. HTTP-based remote transport introduces network identity, TLS, origins,
    authorization, session handling, and exposure to the internet. The protocol describes messages;
    it does not decide which server is trusted or which user may invoke which capability.
    """),
    ("code", r'''
    # Inspect complete discovered schemas rather than only names.
    async def describe_server():
        params = StdioServerParameters(command=sys.executable, args=["demo_mcp_server.py"])
        async with stdio_client(params) as streams:
            async with ClientSession(*streams) as session:
                await session.initialize()
                response = await session.list_tools()
                for tool in response.tools:
                    print({"name": tool.name, "description": tool.description,
                           "inputSchema": tool.inputSchema})
    await describe_server()
    '''),
    ("md", r"""
    ## 24.4 Server engineering and error design

    Tools should be thin adapters over tested domain functions, with typed parameters, bounded
    inputs/outputs, explicit side-effect annotations in descriptions, and sanitized errors. Avoid
    ambient access to an entire filesystem or network. Resolve paths against an allowed root and
    defend against symlinks/traversal. Inject scoped credentials per request/user rather than giving
    a long-lived universal token to the process. Apply timeouts and cancellation to downstream work.

    Resources need stable URI schemes, MIME types, provenance, size limits, and authorization.
    Resource contents are untrusted when inserted into an LLM prompt. Prompt templates should be
    versioned and transparent. Log protocol/tool metadata while redacting secrets and sensitive
    content. Test the server functions directly, then through an MCP client, then through a model
    host; these layers catch different failures.
    """),
    ("code", r'''
    # Directly test the generated server's domain function without an LLM.
    import importlib.util
    spec = importlib.util.spec_from_file_location("demo_mcp_server", "demo_mcp_server.py")
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    for text in ["one two three", "", "word " * 50]:
        print(repr(text[:20]), module.word_count(text))
    try:
        module.word_count("x" * 10_001)
    except Exception as exc:
        print("bounded input test:", type(exc).__name__, str(exc))
    '''),
    ("md", r"""
    ## 24.5 Host bridge and security policy

    The host decides what the model sees and what is executed. Convert discovered schemas to the
    model's tool format, but maintain a server/tool allowlist, validate arguments locally, bind calls
    to the current user, and authorize every operation. Cap tool result bytes/tokens and label their
    origin. Do not automatically install/connect servers suggested by untrusted content. Surface the
    exact server identity and effect when requesting approval.

    Threats include malicious servers, compromised dependencies, tool-name collisions, schema
    deception, indirect prompt injection in resources/results, credential exfiltration, confused-
    deputy actions, cross-tenant access, and denial of service. For local servers, review command,
    arguments, environment variables, binary path, working directory, and filesystem/network scope.
    For remote servers, verify endpoint/authentication and constrain redirects/origins.

    **Deployment reference:** pin SDK/server versions; use a lockfile; isolate process/container;
    least-privilege credentials; authenticate users and servers; tool/resource policy; bounded
    payloads/time; approvals; audit trail; health/restart policy; protocol compatibility tests; and
    an incident path to revoke the connection.
    """),
)

lesson(28,
    ("md", r"""
    ## 28.4 From product contract to evaluation record

    Write requirements as observable claims. “Helpful” is underspecified; “returns the correct
    account policy, cites the effective policy version, and abstains when no policy applies” yields
    separable graders. Define input distribution, expected output/behavior, allowed variation,
    severity, latency/cost constraints, and action after failure. Evaluation examples should store
    stable IDs, inputs/context, references or grading metadata, slices, provenance, and annotation
    status—not only prompt/answer pairs.

    Development examples guide iteration; a frozen test estimates generalization; a shadow set can
    monitor production drift. Prevent contamination by grouping related sources and limiting repeated
    inspection of final tests. Add discovered production failures to a regression set, but also sample
    broadly so metrics do not become a catalog of yesterday's bugs. Version datasets and graders
    independently; a score without both versions is not reproducible.
    """),
    ("code", r'''
    # A compact evaluation-record schema and validation example.
    from pydantic import BaseModel, ConfigDict, Field
    class EvalRecord(BaseModel):
        model_config = ConfigDict(extra="forbid")
        id: str
        input: str
        reference: str | None = None
        slices: list[str] = Field(default_factory=list)
        severity: str = "normal"
        metadata: dict = Field(default_factory=dict)

    record = EvalRecord(id="rag-001", input="What is the refund window?",
        reference="30 days", slices=["answerable", "policy"],
        metadata={"supporting_source": "refund-policy-v3"})
    print(record.model_dump_json(indent=2))
    '''),
    ("md", r"""
    ## 28.5 Grader design and aggregation

    Deterministic graders are preferred when the contract is deterministic: JSON/schema, regex,
    set equality, numeric tolerance, executable unit tests, tool name/arguments, citation IDs, and
    forbidden actions. Text similarity measures lexical/semantic closeness but not correctness.
    Model graders handle nuance at higher variance/bias. Human review handles ambiguity and high
    stakes but needs instructions, training, blinding, overlap, adjudication, and agreement analysis.

    Avoid averaging incommensurate metrics into one opaque score. Use gates for critical invariants,
    then report a metric vector and slices. If a weighted score is necessary, justify weights using
    business cost and show components. Macro averaging weights groups/classes equally; micro averaging
    weights examples/events. Decide how partial credit, multiple references, ties, abstentions, parser
    failures, and grader errors are counted before viewing system results.
    """),
    ("code", r'''
    # Slice report with Wilson intervals for binary outcomes.
    from collections import defaultdict
    def wilson(successes, n, z=1.96):
        if not n: return (float("nan"), float("nan"))
        p = successes/n; den = 1 + z*z/n
        center = (p + z*z/(2*n))/den
        radius = z*((p*(1-p)/n + z*z/(4*n*n))**.5)/den
        return center-radius, center+radius
    outcomes = [("short", 1), ("short", 1), ("short", 0),
                ("long", 1), ("long", 0), ("long", 0)]
    grouped = defaultdict(list)
    for group, value in outcomes: grouped[group].append(value)
    for group, values in grouped.items():
        lo, hi = wilson(sum(values), len(values))
        print(group, f"{sum(values)/len(values):.1%}", f"95% CI [{lo:.1%}, {hi:.1%}]")
    '''),
    ("md", r"""
    ## 28.6 Comparative statistics and experiment discipline

    Evaluate systems on the same examples and use paired differences. Bootstrap examples (or
    independent clusters such as users/documents) to estimate uncertainty. For binary paired
    outcomes, inspect discordant pairs and consider McNemar-style analysis. Multiple experiments
    and metric fishing inflate false discoveries; predeclare primary metrics and retain all results.
    Statistical significance does not imply practical significance—set a minimum meaningful effect.

    Stochastic generation adds within-example variation. Decide whether the estimand is expected
    quality over sampling, pass@k, worst-case, or one production run; repeat seeds accordingly.
    Model-based graders add grader variation and should be sampled/calibrated too. Keep raw outputs
    to rerun new graders without repaying generation cost. Human adjudication should be blind to
    system identity and randomized in presentation order.
    """),
    ("code", r'''
    # Paired bootstrap for the accuracy difference between systems A and B.
    import random, numpy as np
    a = np.array([1,1,0,1,0,1,0,0,1,1])
    b = np.array([1,0,0,1,1,1,0,0,0,1])
    observed = float((a-b).mean())
    rng = random.Random(7)
    deltas = []
    for _ in range(10_000):
        idx = [rng.randrange(len(a)) for _ in a]
        deltas.append(float((a[idx]-b[idx]).mean()))
    lo, hi = np.quantile(deltas, [.025,.975])
    print(f"paired delta={observed:+.3f}; bootstrap 95% CI [{lo:+.3f}, {hi:+.3f}]")
    '''),
    ("md", r"""
    ## 28.7 Evaluation operations reference

    An experiment artifact should include application code commit, model ID/revision/provider,
    prompt/template/tool schemas, decoding and seed, retrieval/index versions, dataset fingerprint,
    grader versions/prompts/models, raw outputs/traces, environment, latency/tokens/cost, and summary
    tables. Cache only when these inputs match. CI uses a small fast critical/regression suite;
    scheduled runs cover broader/stochastic/costly cases; canary/shadow monitoring validates live
    distributions without allowing unreviewed automated actions.

    Common traps: test-set overfitting; leakage; only easy/answerable examples; unlabeled slice gaps;
    references with errors; semantic similarity as factuality; ignoring parser/grader failures;
    changing model and prompt simultaneously; averages without denominators/uncertainty; judging
    only final answer when tools/retrieval can cause harm; and optimizing a proxy after it stops
    tracking user value.

    Use evaluation diagnostically: preserve failure clusters and traces, form a hypothesis, change
    one component, rerun paired examples and broad regressions, then document the decision.
    """),
)

lesson(29,
    ("md", r"""
    ## 29.3 Rubric construction and prompt anatomy

    Criteria should be observable, minimally overlapping, and tied to the decision. Define anchors
    with examples: a correctness 1 contains a material error; 3 is mostly correct with a meaningful
    omission; 5 is fully supported. Specify precedence (critical factual errors override style),
    reference authority, allowed assumptions, abstention handling, and whether brevity is required.
    Ask for decisive evidence before/alongside a score but avoid eliciting private chain-of-thought;
    a concise justification citing answer spans is enough for auditing.

    Delimit question/reference/candidate as data, randomize candidate labels, hide model identities,
    and constrain output. Position the rubric before examples consistently. Few-shot examples improve
    calibration but can anchor style and leak content; select representative boundary cases. Long
    judge prompts can bury criteria. Version the entire judge template, schema, model/revision,
    provider, and decoding parameters.
    """),
    ("code", r'''
    # Validate and normalize judge results before aggregation.
    def validate_judgment(result):
        if set(result.get("scores", {})) != set(RUBRIC):
            raise ValueError("criterion mismatch")
        for criterion, score in result["scores"].items():
            if not isinstance(score, int) or not 1 <= score <= 5:
                raise ValueError(f"invalid {criterion} score")
        if not isinstance(result.get("critical_error"), bool):
            raise ValueError("critical_error must be boolean")
        return {**result, "weighted": sum(
            result["scores"][k] * spec["weight"] for k, spec in RUBRIC.items())}

    mock = {"scores": {k: 4 for k in RUBRIC}, "critical_error": False,
            "evidence": "Candidate addresses the reference directly."}
    print(validate_judgment(mock))
    '''),
    ("md", r"""
    ## 29.4 Pointwise, pairwise, listwise, and reference-free judging

    Pointwise scoring gives absolute-looking values but judges use implicit standards and score
    distributions can drift. Pairwise comparison is cognitively simpler and often more reliable,
    but yields relative preferences and suffers position bias. Listwise ranking increases context
    and ordering interactions. Reference-based judging helps objective tasks only when the reference
    is correct/complete. Reference-free judging is necessary for open-ended outputs but depends more
    heavily on rubric and judge knowledge.

    For pairwise evaluation, run A/B and B/A. Map swapped labels back to systems. Consistent A/B wins
    are usable; inconsistent decisions should be ties/uncertain or human-routed, not arbitrarily one
    side. Randomize order across the dataset. Control length by rubric rather than truncating answers.
    When comparing many systems, balanced tournament designs or rating models can reduce calls, but
    transitivity is not guaranteed.
    """),
    ("code", r'''
    # Reconcile forward and swapped pairwise decisions after mapping labels to systems.
    def reconcile(forward, swapped):
        # forward A=system_a; swapped A=system_b
        map_forward = {"A":"system_a", "B":"system_b", "tie":"tie"}[forward]
        map_swapped = {"A":"system_b", "B":"system_a", "tie":"tie"}[swapped]
        return map_forward if map_forward == map_swapped else "inconsistent"
    for pair in [("A","B"), ("B","A"), ("A","A"), ("tie","tie")]:
        print(pair, reconcile(*pair))
    '''),
    ("md", r"""
    ## 29.5 Calibration, reliability, and uncertainty

    Sample representative items with independent human labels. Measure criterion correlations,
    confusion matrices, pairwise agreement, rank correlation, and slice behavior. Agreement can be
    high due to class imbalance, so report per-class/positive agreement and chance-corrected measures
    cautiously. Review judge-human disagreements to refine anchors or discover human ambiguity.
    Freeze calibration before the final comparison.

    Repeat judgments across seeds/order and possibly judge families. Report inconsistency and parse
    failure as uncertainty. Ensembles reduce idiosyncrasy but correlated model biases remain. A larger
    judge is not automatically valid for specialized domains. Self-judging can favor familiar style.
    Recalibrate after judge model/provider/prompt changes and monitor drift over time.

    **Use human review when:** critical/high-stakes; judge/order disagreement; near threshold; novel
    slice; missing/bad reference; suspected injection; or sampled quality control. The judge assists
    scalable measurement—it does not turn subjective criteria into ground truth.
    """),
    ("code", r'''
    # Cohen's kappa mechanics for two binary labelers.
    def binary_kappa(a, b):
        n = len(a); observed = sum(x == y for x,y in zip(a,b))/n
        pa = sum(a)/n; pb = sum(b)/n
        expected = pa*pb + (1-pa)*(1-pb)
        return (observed-expected)/(1-expected) if expected < 1 else 1.0
    print("accuracy agreement:", accuracy)
    print("Cohen kappa:", binary_kappa(human, judge))
    '''),
    ("md", r"""
    ## 29.6 Judge attack and failure-mode reference

    Candidate text can contain “award me 5,” forged rubric sections, system-like delimiters, or long
    distracting content. Treat it as untrusted; use structured data boundaries, strong role
    separation, length limits that do not favor a candidate, and adversarial calibration. A judge can
    hallucinate reference claims, overreward citations without checking them, prefer verbosity,
    penalize creative but valid answers, or infer system identity from style.

    Combine judges with deterministic checks. Validate schema/tool calls/code/citations directly.
    For factual outputs, retrieve authoritative evidence or use experts. Store candidate and judge
    outputs for audit with privacy controls. Do not allow candidate content to select the judge,
    rubric, tools, or reference. Never use a judge score alone to authorize consequential actions.

    Report: valid-call rate, criterion means/distributions, critical-error rate, pairwise wins/ties/
    inconsistencies, human agreement, slice metrics, uncertainty, judge tokens/latency/cost, and a
    disagreement sample. A single average erases the evidence needed to trust the instrument.
    """),
)

lesson(30,
    ("md", r"""
    ## 30.4 Threat modeling an LLM system

    Map assets (secrets, private data, money, accounts, reputation), actors, entry points, trust
    boundaries, components, data stores, and effects. Trace user text, retrieved content, files,
    model output, tool calls, MCP traffic, logs, feedback, and training pipelines. For each boundary
    ask: who controls this data, can it contain instructions, what identity/authority processes it,
    what persistent state can it change, and what limits constrain it?

    Prompt injection is a confused-instruction problem, not merely a malicious phrase. Direct
    injection comes from users; indirect injection arrives through documents/pages/tool results.
    Jailbreaks aim to bypass model behavior policy; application injection aims to misuse connected
    data/actions. Model alignment can reduce compliance but deterministic application controls must
    protect assets even when the model is fully compromised.
    """),
    ("code", r'''
    # Represent a compact threat register that can become test cases.
    threats = [
        {"id":"T1", "source":"retrieved document", "threat":"indirect injection",
         "asset":"email tool", "control":"read-only scope + approval", "test":"no send call"},
        {"id":"T2", "source":"tool argument", "threat":"SSRF",
         "asset":"internal network", "control":"URL allowlist", "test":"metadata IP blocked"},
        {"id":"T3", "source":"tenant ID", "threat":"cross-tenant access",
         "asset":"private records", "control":"server-side identity binding", "test":"foreign ID denied"},
    ]
    for threat in threats: print(threat)
    '''),
    ("md", r"""
    ## 30.5 Authorization, isolation, and side effects

    Bind user identity and tenant server-side; never let the model choose an unrestricted account ID.
    Give tools narrowly scoped credentials. Split read from write and high-risk operations. Validate
    resource ownership at execution time. Use network egress allowlists/proxies, filesystem roots,
    sandboxed code, database parameterization, and output escaping. Assume the model can construct
    adversarial strings for every downstream interpreter.

    Human approval must present the exact normalized action and consequences after validation, not a
    model summary. Approval is specific, fresh, and cannot expand scope. Make writes idempotent and
    auditable; provide reversible drafts where possible. Limit steps, requests, tokens, execution time,
    bytes, recursion, and spend. Bound queues to prevent resource exhaustion. Separate environments
    and tenants; encrypt data; minimize retention; redact traces while keeping security evidence.
    """),
    ("code", r'''
    # Server-side tenant binding: requested tenant is never authority by itself.
    def fetch_record(authenticated_tenant, requested_tenant, record_id, database):
        if requested_tenant != authenticated_tenant:
            raise PermissionError("cross-tenant request denied")
        key = (authenticated_tenant, record_id)
        if key not in database: raise KeyError("record not found")
        return database[key]

    db = {("acme", "1"): {"value": "private"}, ("other", "1"): {"value": "secret"}}
    print(fetch_record("acme", "acme", "1", db))
    try: fetch_record("acme", "other", "1", db)
    except Exception as exc: print(type(exc).__name__, exc)
    '''),
    ("md", r"""
    ## 30.6 RAG, agent, MCP, and training-specific threats

    RAG can retrieve poisoned/injected sources, expose unauthorized chunks, cite stale content, or
    leak private text through embeddings/logs. Enforce ACL filters, source trust/provenance, deletion,
    and context labeling. Agents add iterative amplification, unauthorized tools, loops, and
    side-effects. MCP adds supply-chain/server identity, local process privileges, remote auth, and
    resource injection. Multimodal systems can hide instructions in images or metadata.

    Training pipelines face dataset poisoning, benchmark contamination, private-data memorization,
    malicious model artifacts/custom code, and compromised dependencies. Pin/review provenance,
    scan artifacts, prefer Safetensors, isolate untrusted code, audit data, and evaluate canaries or
    memorization risk. Fine-tuning does not reliably remove knowledge from a base model. Model output
    moderation and data loss prevention are separate controls with different failure modes.
    """),
    ("code", r'''
    # Security invariants are deterministic pass/fail facts over an execution trace.
    trace = [
        {"type":"retrieve", "source":"policy-v3", "tenant":"acme"},
        {"type":"tool", "name":"lookup_policy", "side_effect":False},
        {"type":"answer", "contains_secret":False},
    ]
    invariants = {
        "no_side_effect": not any(e.get("side_effect") for e in trace),
        "single_tenant": all(e.get("tenant", "acme") == "acme" for e in trace),
        "no_secret": not any(e.get("contains_secret") for e in trace),
    }
    print(invariants, "release gate:", all(invariants.values()))
    '''),
    ("md", r"""
    ## 30.7 Security testing and incident readiness reference

    Turn the threat register into automated adversarial cases. Vary encoding, languages, document
    position, role-like syntax, nested files, redirects/DNS/IP forms, tool outputs, multi-turn setup,
    and long-context distraction. Assert invariants from traces and real effects—not that the model
    said it refused. Run tests after model/prompt/tool/retriever/dependency changes and conduct human
    red teams for novel chains. Keep a safe isolated environment with synthetic assets.

    Monitor authorization denials, unusual tool sequences, repeated failures, high token/spend,
    unknown destinations, cross-tenant attempts, injection detections, secret/DLP alerts, and model/
    server changes. Logs themselves are sensitive. Prepare kill switches: revoke credentials, disable
    tools/servers, block destinations, stop deployments, invalidate sessions, and remove poisoned
    sources. Document owners and communication.

    Security is residual-risk management. State which attacks controls prevent, detect, or merely
    reduce; record assumptions; retest them. “The model usually refuses” is neither an authorization
    control nor an incident plan.
    """),
)

lesson(32,
    ("md", r"""
    ## 32.4 Vision-language architecture families

    Many VLMs encode images into patch features, project them into the language model's hidden
    space, and place visual tokens alongside text tokens. Others use cross-attention or resampling
    modules to compress visual features. Native multimodal models train integrated representations.
    The processor decides resize, aspect handling, crops/tiles, normalization, placeholder count,
    and chat formatting; model and processor revisions must match.

    A square image with patch size p yields roughly `(H/p)*(W/p)` raw patches before model-specific
    compression or tiling. Dynamic-resolution systems may create many more tokens for large or
    unusual-aspect images. Multiple images multiply prefill/memory. Resize can erase small text;
    aggressive tiling preserves it at cost. Report original/transformed dimensions and visual token
    counts when debugging rather than assuming the model “saw” the source.
    """),
    ("code", r'''
    # Patch-count planning intuition (actual processors may tile/compress differently).
    import math
    def patch_count(width, height, patch=14):
        return math.ceil(width/patch) * math.ceil(height/patch)
    for width, height in [(224,224), (640,480), (1920,1080), (2480,3508)]:
        print(f"{width}x{height}: ~{patch_count(width,height):,} raw {14}x{14} patches")
    '''),
    ("md", r"""
    ## 32.5 OCR, layout, charts, and grounded extraction

    OCR recognizes glyphs; document understanding also needs reading order, key-value association,
    tables, page references, and visual hierarchy. Native PDF extraction is often cheaper and exact
    for embedded text; OCR handles scans; layout models/VLMs interpret structure. A robust pipeline
    combines them and retains page/bounding-box provenance. Compare extracted text to rendered pages
    because hidden text layers can be wrong.

    For charts, distinguish perception (axes, legend, marks, labels) from reasoning (trend, comparison,
    calculation). Ask for structured fields with evidence coordinates or quoted labels. Validate
    numeric units and recompute derivable quantities. For forms/invoices, define normalized schema,
    required/optional fields, locale-aware dates/currency, confidence/routing, and page/region evidence.
    Do not accept fluent summaries as proof of exact field extraction.
    """),
    ("code", r'''
    # Example extraction schema and deterministic semantic validation.
    from pydantic import BaseModel, Field, model_validator
    class InvoiceExtraction(BaseModel):
        invoice_id: str
        currency: str = Field(pattern=r"^[A-Z]{3}$")
        subtotal: float = Field(ge=0)
        tax: float = Field(ge=0)
        total: float = Field(ge=0)
        evidence_page: int = Field(ge=1)
        @model_validator(mode="after")
        def totals_match(self):
            if abs((self.subtotal + self.tax) - self.total) > .02:
                raise ValueError("subtotal + tax does not match total")
            return self
    print(InvoiceExtraction(invoice_id="INV-7", currency="USD", subtotal=100,
                            tax=23.45, total=123.45, evidence_page=1))
    '''),
    ("md", r"""
    ## 32.6 Multimodal prompting, batching, and serving

    Follow the model's exact chat template and content-part structure. State which image each
    instruction refers to; order images deterministically; avoid ambiguous “above/below.” Separate
    user instructions from document content because images can contain prompt injection. Request a
    schema and concise evidence, then validate. Cropping targeted regions can improve small-text
    accuracy, but a cropper must not omit context; retain original coordinates.

    Batch by compatible image size/token count to limit padding. Preprocessing can become CPU-bound;
    profile decode, resize, OCR, transfer, prefill, and generation separately. Cache safe deterministic
    preprocessing. Limit image count, pixels, file bytes, pages, decompression ratio, and processing
    time. Strip or handle metadata intentionally. Serving support varies by vLLM/TGI/model version;
    test exact modalities, templates, quantization, and concurrent memory.
    """),
    ("code", r'''
    # Preserve crop provenance when zooming into a region.
    box = (40, 70, 610, 175)  # left, top, right, bottom in original pixels
    crop = image.crop(box)
    display(crop.resize((crop.width*2, crop.height*2)))
    provenance = {"original_size": image.size, "crop_box_xyxy": box,
                  "scale_for_display": 2}
    print(provenance)
    '''),
    ("md", r"""
    ## 32.7 Multimodal evaluation and safety reference

    Build representative source files, not screenshots chosen for demos. Field extraction uses exact/
    normalized accuracy and per-field precision/recall; OCR uses character/word error rate; tables use
    structural/cell metrics; grounding uses IoU/pointing; charts use numeric tolerance and reasoning
    correctness; descriptions require rubric/human review. Slice by resolution, font, scan quality,
    rotation, handwriting, language, layout, page count, chart type, color dependence, and adversarial
    text. Evaluate abstention when content is illegible or absent.

    Threats include visual prompt injection, QR/links, steganographic or tiny text, malicious file
    parsers, decompression bombs, metadata leakage, faces/biometrics, copyrighted/private documents,
    and cross-tenant caches. Sandbox parsers, verify MIME/signature, cap resources, apply malware/DLP
    policy, preserve access controls, and require confirmation before image-derived instructions cause
    actions. The model should describe evidence; deterministic code should authorize effects.
    """),
)

lesson(33,
    ("md", r"""
    ## 33.4 Latency and throughput decomposition

    End-to-end latency includes client/network, admission queue, tokenization/rendering, retrieval/
    tools, prefill, decode, validation, and streaming transport. TTFT includes everything until the
    first visible delta. Time per output token/inter-token latency describes decode smoothness. Report
    p50/p95/p99 by prompt/output length and workload class; averages hide tail behavior. Throughput
    can mean requests/sec or input/output/total tokens/sec—state which.

    Little's Law relates average concurrency \(L\), arrival rate \(\lambda\), and time in system
    \(W\): \(L=\lambda W\) under stable conditions. As utilization nears capacity, queueing rises
    sharply. Maximize sustainable throughput subject to latency/error/quality SLO, not benchmark peak.
    Separate cold start/model load from warm requests and distinguish provider-reported usage from
    locally estimated tokens.
    """),
    ("code", r'''
    # Summarize a synthetic latency trace by percentile and stage.
    import numpy as np
    trace_rows = [
        {"queue":10,"prefill":80,"decode":240}, {"queue":15,"prefill":90,"decode":260},
        {"queue":180,"prefill":100,"decode":310}, {"queue":12,"prefill":75,"decode":220},
        {"queue":400,"prefill":120,"decode":360},
    ]
    total = np.array([sum(r.values()) for r in trace_rows])
    for p in [50, 95, 99]: print(f"p{p} total={np.percentile(total,p):.1f} ms")
    print("mean stage ms:", {k: np.mean([r[k] for r in trace_rows]) for k in trace_rows[0]})
    '''),
    ("md", r"""
    ## 33.5 Resilience patterns and their boundaries

    Set a total deadline, then allocate stage timeouts; otherwise retries can exceed the user's
    budget. Retry transient rate-limit/network/5xx failures only when safe. Respect retry-after, use
    exponential backoff with jitter, cap attempts, and preserve idempotency. Hedged requests may lower
    tails but multiply cost/load and need cancellation. Circuit breakers prevent hammering a failing
    dependency but require careful half-open recovery. Bulkheads isolate workloads so one tenant or
    slow tool cannot exhaust all concurrency.

    Backpressure begins before overload: bounded admission queues, per-tenant quotas, concurrency and
    token budgets, and load shedding by priority. Streaming disconnects/cancellations should propagate
    upstream. Fallbacks are product behavior, not merely infrastructure: switching to a smaller model,
    cached answer, retrieval-only response, or abstention must meet quality/safety evals and be visible
    in traces. Never retry deterministic validation or authorization failures.
    """),
    ("code", r'''
    # Deadline budgeting helper: remaining time shrinks across stages/retries.
    import time
    class Deadline:
        def __init__(self, seconds): self.ends = time.monotonic() + seconds
        def remaining(self): return max(0.0, self.ends - time.monotonic())
        def timeout(self, stage_cap):
            value = min(stage_cap, self.remaining())
            if value <= 0: raise TimeoutError("request deadline exhausted")
            return value
    deadline = Deadline(2.0)
    print("retrieval timeout", deadline.timeout(.4))
    time.sleep(.02)
    print("generation timeout", deadline.timeout(1.5))
    '''),
    ("md", r"""
    ## 33.6 Observability, SLOs, and privacy

    Metrics aggregate health; logs capture discrete events; traces connect stages; profiles explain
    resource time. Use low-cardinality metric labels—never user IDs or prompts. Trace IDs correlate
    request/model/retrieval/tool spans. Capture model/revision, prompt version (not necessarily text),
    decoding, token usage, cache, finish reason, validation, retries/fallback, and sampled quality.
    Redact at collection, restrict access, encrypt, set retention, and give tenants deletion controls.

    Define SLIs and SLOs by user experience: successful validated responses under a latency bound,
    availability excluding invalid requests, quality on sampled labeled traffic, and budget. Error
    budgets govern release pace. Alert on actionable symptoms and burn rates rather than every transient
    error. Dashboards slice by model/version/route/length/tenant tier while controlling cardinality.
    Correlate quality with latency/cost; a fast invalid response is not successful.
    """),
    ("code", r'''
    # Error-budget calculation for a 99.5% monthly availability SLO.
    minutes = 30 * 24 * 60
    slo = .995
    budget_minutes = minutes * (1-slo)
    consumed = 95
    print(f"monthly error budget: {budget_minutes:.1f} min")
    print(f"consumed: {consumed/budget_minutes:.1%}; remaining: {budget_minutes-consumed:.1f} min")
    '''),
    ("md", r"""
    ## 33.7 Load testing and release engineering reference

    Model prompt/output distributions, arrival process, streaming, cancellations, retrieval/tools,
    cache hit rates, tenants, and failures. Open-loop tests send arrivals independently and expose
    queueing; closed-loop clients wait for responses and can hide overload through coordinated
    omission. Warm up kernels/caches, run long enough for steady state, and record hardware/software.
    Find the knee where tail latency or errors accelerate, then operate with headroom.

    Release artifacts pin code, model, tokenizer/template, prompt, index, config, and dependencies.
    Run offline evals and load tests; deploy canary; compare quality/latency/cost/safety; expand
    gradually; retain automatic/manual rollback. Schema migrations and cache compatibility need plans.
    Test cold start, dependency outage, rate limits, corrupt responses, cancellation, deploy during
    load, and rollback—not only happy-path throughput.

    Capacity plan in tokens: offered input/output tokens/sec, concurrency, KV-cache demand, replicas,
    autoscaling lag, quotas, and cost. Autoscaling on CPU alone misses GPU/KV/queue saturation.
    """),
)

lesson(36,
    ("md", r"""
    ## 36.7 Ollama deployment decision record

    Before adopting an engine, write down the workload and constraint that selected it. Ollama is a
    strong default for a developer workstation, an offline demonstration, or a single-user application
    where operational simplicity matters more than accelerator-wide throughput. Revisit that decision
    when concurrency, latency objectives, model size, multimodal requirements, tenant isolation, adapter
    density, or observability needs change. Migration is easiest when prompts use correct chat templates,
    business logic is outside the engine, and a narrow internal client contract is backed by conformance
    tests against every supported server.

    Keep an inventory of downloaded models, digests, provenance, license, size, quantization, owners, and
    last use. Local weights consume material disk and may have redistribution restrictions. Updating the
    Ollama application or a mutable model tag is a release: stage it, rerun frozen quality and performance
    tests, and retain a rollback path. Data locality is a valuable property, but only a documented threat
    model can establish which processes, users, plugins, tools, logs, and network destinations can still
    observe a request.
    """),
    ("code", r'''
    engine_decision = {
        "Ollama": "local simplicity, privacy-sensitive prototypes, modest concurrency",
        "vLLM": "high-throughput accelerator serving and continuous batching",
        "Transformers": "research control, architecture debugging, direct Python integration",
        "managed endpoint": "outsourced infrastructure lifecycle and scaling",
    }
    for engine, fit in engine_decision.items(): print(f"{engine:16} | {fit}")
    ''') ,
)

lesson(37,
    ("md", r"""
    ## 37.5 vLLM engine mental model

    Requests pass through an API frontend/tokenizer into an engine scheduler. Prefill computes prompt
    states; decode advances active sequences token by token. A block manager allocates logical KV-cache
    blocks to physical memory, reducing fragmentation and enabling operations such as prefix sharing.
    Continuous batching changes the active batch each iteration. Sampling, stop conditions, streaming,
    and request cancellation interact with scheduler state.

    The model's architecture determines tensor shapes, KV heads, supported precision/quantization,
    multimodal inputs, and tool/reasoning parsers. The engine version determines kernels and API
    behavior. OpenAI compatibility is a client protocol surface, not proof every OpenAI parameter or
    model capability behaves identically. Read the current compatibility/serve help for the installed
    release and pin it.
    """),
    ("code", r'''
    # Capacity worksheet for weights plus KV cache (planning estimate only).
    def serving_memory_gib(params_b, weight_bits, layers, kv_heads, head_dim,
                           cached_tokens, kv_bits=16, workspace_fraction=.15):
        weights = params_b * 1e9 * weight_bits/8
        kv = 2 * layers * kv_heads * head_dim * cached_tokens * kv_bits/8
        subtotal = weights + kv
        return {"weights":weights/2**30, "kv":kv/2**30,
                "with_workspace_margin":subtotal*(1+workspace_fraction)/2**30}
    print(serving_memory_gib(7, 16, 32, 8, 128, cached_tokens=32_768*8))
    print("cached_tokens is total across concurrent sequences, not max context alone.")
    '''),
    ("md", r"""
    ## 37.6 Server configuration and compatibility

    Important controls include served model name/revision, tokenizer/chat template, dtype,
    quantization, maximum model length, GPU memory utilization, maximum batched tokens/sequences,
    prefix caching, speculative decoding, tensor/data/pipeline/expert parallelism, tool/reasoning
    parser, structured-output backend, and logging/metrics. Defaults evolve. Save the exact command or
    YAML config and `vllm --version`; CLI flags override config according to documented precedence.

    A chat endpoint needs a valid chat template. Tool calls require model formatting plus a matching
    parser and application loop. Reasoning models may need a reasoning parser. Structured output
    constrains generation but can reduce throughput or reject unsupported schemas. Quantized formats
    need compatible kernels/hardware and quality evals. LoRA serving changes memory/routing and dynamic
    adapter loading is a security/operations decision, not a default internet-facing feature.
    """),
    ("code", r'''
    # Produce a reviewable serve command from explicit settings—do not execute in this notebook.
    config = {
        "model":"Qwen/Qwen2.5-1.5B-Instruct", "host":"127.0.0.1", "port":8000,
        "dtype":"auto", "max-model-len":8192, "gpu-memory-utilization":0.90,
        "enable-prefix-caching":True,
    }
    command = ["vllm", "serve", config.pop("model")]
    for key, value in config.items():
        if value is True: command.append(f"--{key}")
        elif value is not False: command.extend([f"--{key}", str(value)])
    print(" ".join(command))
    '''),
    ("md", r"""
    ## 37.7 Parallelism, replicas, and topology

    Tensor parallelism shards layer matrices and introduces frequent collectives; keep it within a
    fast-connected node when possible. Pipeline parallelism splits layers/stages but can introduce
    bubbles. Data parallelism replicates the model and distributes requests, increasing aggregate
    throughput and fault isolation when each replica fits. Expert parallelism distributes MoE experts.
    Multi-node setups need explicit launch/network/NCCL/Ray or other executor configuration and failure
    handling. More GPUs can be slower when communication dominates.

    Choose the smallest parallel group that makes one replica meet single-request latency and memory,
    then add replicas for traffic when possible. Load balancing should consider queue/cache locality,
    not blind round robin. Prefix-cache benefits disappear if identical prefixes scatter. Autoscale on
    queueing/token load with enough model-load lead time. Keep spare capacity for failures and deploys.
    Benchmark topology with real prompt/output lengths and concurrency.
    """),
    ("code", r'''
    # Simple throughput/capacity planning table.
    offered_rps = 4
    avg_input, avg_output = 800, 200
    offered_tokens = offered_rps * (avg_input + avg_output)
    for replicas in [1,2,4]:
        capacity_per_replica = 1800  # measured total tokens/s example
        utilization = offered_tokens / (replicas*capacity_per_replica)
        print(replicas, f"offered utilization={utilization:.1%}",
              "headroom_ok" if utilization < .7 else "queue risk")
    '''),
    ("md", r"""
    ## 37.8 Benchmark methodology and optimization

    Separate offline throughput from online latency benchmarks. Use representative prompt/output
    length distributions and arrival patterns. Warm up. Report TTFT, inter-token latency, end-to-end
    p50/p95/p99, input/output tokens/sec, request throughput, queue time, errors, aborts, GPU utilization,
    memory/KV occupancy, and prefix hit rate. Confirm generated quality and token counts; a configuration
    that truncates or emits shorter answers appears artificially fast.

    Optimization order: establish correctness; size context/admission; choose supported dtype/kernels;
    enable prefix caching for repeated prefixes; tune batching limits against tail latency; evaluate
    quantization; consider speculative decoding when draft acceptance and workload justify it; then
    evaluate parallelism/replicas. Change one factor at a time. CUDA graphs/compilation and kernel
    selection can have shape-dependent tradeoffs. Record power/cost if economics matter.
    """),
    ("md", r"""
    ## 37.9 Production and security reference

    Put the server behind authenticated TLS ingress with per-tenant rate/token/request-size limits.
    Bind admin/dev endpoints privately. Do not use a shared example API key. Control model/tokenizer
    downloads, remote code, adapter loading, filesystem/cache permissions, and network egress. Prompts
    and outputs are sensitive; configure logs intentionally. Patch engine/model dependencies and scan
    images. Validate structured/tool output in the application.

    Readiness should wait for loaded/warmed service; liveness should not restart merely because the GPU
    is busy. Scrape metrics and correlate request IDs. Graceful shutdown drains/aborts according to
    policy. Test OOM recovery, worker/GPU failure, corrupt request, overload, client cancellation,
    rolling deploy, model-load failure, and rollback. Pin image digest, vLLM version, model commit,
    tokenizer/template, config, driver/CUDA, and hardware in the release record.

    Colab is suitable for client experiments and some temporary single-GPU exploration, not a durable
    multi-user vLLM service. Use controlled Linux GPU infrastructure for serving benchmarks and
    production conclusions.
    """),
)

# Compact, lesson-specific reference appendices. These are intentionally placed before each
# notebook's exercises so the notebook works both as a guided lesson and a later lookup resource.
REFERENCE_APPENDICES = {
1: r"""
## 1.7 Tokenization reference and common pitfalls

| Term | Practical meaning |
|---|---|
| Vocabulary | ID-to-piece inventory paired with embedding/output rows |
| Encode/decode | Text→IDs and IDs→text; round-trip behavior may normalize details |
| Special token | Control token with template/model semantics, not ordinary prose |
| Chat template | Deterministic rendering of structured messages into model tokens |
| Attention mask | Controls visible/valid input positions |
| Label mask | Controls which target positions contribute loss, commonly with `-100` |
| Packing | Combining examples to reduce padding waste |

Common failures are counting tokens before applying the chat template, adding a pad/special token
without resizing/training embeddings, using a tokenizer from a similarly named but different model,
double-adding BOS/EOS, decoding the prompt together with generated output, and silently truncating
from the wrong side. Unicode adds subtleties: visually identical strings can use different code-point
normalization; emoji can contain joiners and modifiers; byte fallback preserves inputs but may be
token-inefficient. Log exact token IDs when a format behaves unexpectedly.

**Connection forward:** token length drives training batches, attention cost, KV memory, RAG chunk
sizes, and serving capacity. The tokenizer is not preprocessing plumbing—it is the discrete interface
on which every later notebook depends.
""",
2: r"""
## 2.7 Mathematical reference

| Quantity | Definition | Interpretation |
|---|---|---|
| Softmax | \(p_i=e^{z_i}/\sum_j e^{z_j}\) | Categorical distribution from logits |
| NLL | \(-\log p_y\) | Surprise assigned to observed target |
| Cross-entropy | \(H(q,p)=-\sum q\log p\) | Expected NLL under target distribution |
| Entropy | \(H(p)=-\sum p\log p\) | Distribution uncertainty |
| KL | \(D_{KL}(q\|p)=H(q,p)-H(q)\) | Asymmetric discrepancy |
| Perplexity | \(e^{\text{mean NLL}}\) | Effective branching factor under one protocol |

Use natural logs for nats; divide by `ln(2)` for bits. A probability of zero on an observed event
has infinite NLL, motivating stable finite computations. Averaging logits before loss is not the same
as averaging losses. Never softmax before `cross_entropy`, which expects logits. A lower batch loss
can reflect more padding masked, easier/shorter data, or a changed tokenizer rather than learning.

For gradient checks, compare autograd with finite differences on a tiny float64 model. For training
diagnostics, distinguish loss scale, gradient norm, and actual parameter update after optimizer
adaptation. Report evaluation reduction and denominator explicitly.
""",
3: r"""
## 3.6 Decoder architecture reference

| Component | Input/output | Primary role |
|---|---|---|
| Embedding | IDs `[B,T]` → `[B,T,D]` | Learned token representation |
| Norm | `[B,T,D]` → same | Controls scale/optimization |
| Attention | same → same | Communicates among causally visible positions |
| MLP/SwiGLU | same → same | Per-position nonlinear feature transformation |
| Residual | same + same | Identity/gradient path and feature accumulation |
| LM head | `[B,T,D]` → `[B,T,V]` | Vocabulary logits |

Debug in this order: verify shift/mask; overfit one batch; compare logits for a prefix alone versus
the same prefix followed by hidden future tokens; test padding invariance; inspect activation and
gradient norms by layer; compare train/eval modes; then optimize kernels. A model that cannot overfit
a tiny deterministic sequence has an implementation or optimization problem, not insufficient data.

Architectural configuration is part of checkpoint compatibility. Width, heads/KV heads, head
dimension, layer count, MLP width/activation, norm type/epsilon, RoPE configuration, vocabulary,
tie policy, and attention pattern must match weights. Shape-compatible changes can still silently
destroy behavior.
""",
4: r"""
## 4.7 Positional-method reference

| Method | Injected where | Learned? | Extrapolation considerations |
|---|---|---:|---|
| Learned absolute | Added to hidden state | Yes | Table/range-bound |
| Sinusoidal | Added to hidden state | No | Defined beyond training; behavior not guaranteed |
| Relative bias | Attention logits | Often | Bucketing/saturation policy matters |
| ALiBi | Attention logits | Slopes often fixed | Distance penalty supports longer indices |
| RoPE | Rotates Q/K | No frequencies, sometimes scaling config | Phase/frequency distribution shifts |

Do not conflate an implementation accepting longer position IDs with a model reasoning effectively at
that length. Verify tokenizer/model maximums, RoPE cache growth, position IDs under padding/packing,
KV memory, and attention backend support. Changing RoPE base or scaling after training is an inference
intervention and deserves held-out quality tests across lengths and positions.

Useful diagnostics include joint-shift invariance, norm preservation, cached-versus-full logits,
retrieval by depth, multi-hop evidence separation, long-output stability, and perplexity versus token
position. Always compare against a shorter-context/RAG baseline with equal answer evidence.
""",
5: r"""
## 5.7 Efficient-attention decision table

| Technique | Changes mathematical attention? | Main benefit | Main tradeoff |
|---|---:|---|---|
| FlashAttention/SDPA fused kernel | No | Less HBM IO/intermediate memory | Hardware/shape/kernel constraints |
| GQA/MQA | Architecture changes K/V heads | Smaller cache and decode bandwidth | Must be trained/conversion-quality tested |
| Sliding window | Yes, restricts connections | Lower long-context work/cache | Loses direct distant attention |
| Prefix cache | No model change | Reuses identical prompt KV | Hit rate, memory, routing complexity |
| Quantized/offloaded KV | Numeric/storage change | More cached tokens | Conversion, quality, transfer latency |
| Speculative decoding | Same target distribution if implemented exactly | Faster decode with accepted drafts | Draft cost and acceptance dependence |

Never report “attention speed” without batch, Q/K lengths, heads, head dimension, dtype, causal/mask,
device, backend, warmup, forward/backward, and synchronization. Training and decode are different
benchmarks. A memory-efficient kernel may allow a larger batch that improves throughput even if a
single operation is not much faster.

When serving, capacity is total active cached tokens across sequences. Apply admission control before
the allocator reaches failure; leave workspace/deployment headroom; monitor rather than infer cache
health from GPU utilization alone.
""",
7: r"""
## 7.6 Training-data reference

| Stage | Required evidence |
|---|---|
| Source | Provenance, license/consent, timestamp, immutable ID |
| Cleaning | Transformation/filter versions and before/after samples |
| Deduplication | Normalization, exact/near-duplicate method, cluster IDs |
| Split | Group/time/source policy and leakage audit |
| Formatting | Target chat template/tokenizer revision |
| Tokenization | Length distribution, truncation, special tokens |
| Labels | Assistant/role masks and target-token percentage |
| Packing | Boundary/EOS, attention and position policy, utilization |

Data bugs often look like optimizer bugs. Inspect the actual collated batch: decoded IDs, attention
mask, labels with ignored tokens, position IDs, and shifted targets. Calculate tokens/update and
source mixture after sampling—not only raw dataset proportions. Keep a small “golden batch” whose
rendered tokens and masks are regression-tested whenever tokenizer/template/collator versions change.

Privacy and deletion requirements propagate into derived chunks, tokenized caches, checkpoints, and
logs. Dataset documentation should state known gaps and filtering harms, not only row counts.
""",
12: r"""
## 12.6 Optimization reference

| Symptom | First checks |
|---|---|
| Loss unchanged | Labels/mask, trainable params, LR, optimizer step, detached graph |
| Immediate NaN/Inf | Input values, LR, precision/scale, invalid targets, normalization |
| Periodic spikes | Batch length/source, accumulation boundary, scheduler, bad records |
| Train improves, validation worsens | Leakage-free split, overfit, distribution, regularization |
| Resume diverges | RNG, sampler/data position, optimizer/scheduler/scaler state |
| Slow step | Padding/length, dataloader, checkpoint recompute, kernel/device transfer |

Effective global examples/update = per-device microbatch × accumulation × data-parallel replicas;
effective tokens/update also depends on non-padding lengths. If accumulation windows contain different
token counts, dividing every microbatch equally is not identical to a true token-weighted large batch.
Decide and implement the desired normalization.

Clip global norm after unscaling. Log pre-clip norm and proportion clipped. Exclude norm/bias from
weight decay according to an explicit parameter-group rule. Validate optimizer/scheduler step counts
against accumulation; an off-by-K schedule silently changes training.
""",
13: r"""
## 13.7 Memory-efficiency reference

| Method | Saves | Costs/constraints |
|---|---|---|
| Smaller microbatch | Activations | More accumulation/less device utilization |
| Shorter sequence | Activations and attention/cache | May truncate necessary evidence |
| BF16/FP16 | Weights/activations/gradients | Numeric/hardware considerations |
| Activation checkpointing | Saved activations | Forward recomputation |
| FlashAttention | Attention intermediates/IO | Backend eligibility |
| LoRA | Trainable gradients/optimizer states | Frozen-base compute remains |
| QLoRA | Frozen base weights | Quantization kernels/quality |
| FSDP/ZeRO | Sharded states | Communication/complex checkpointing |

Apply techniques based on measured dominant memory. Do not stack every optimization blindly: some
interact, introduce unsupported kernels, or reduce throughput more than an alternative batch/length
choice. After an OOM, a notebook runtime may retain tensors in exceptions/history; delete references,
run garbage collection, empty device cache where applicable, or restart before comparing.

Distributed scaling correctness checks include identical data weighting, synchronized updates, no
duplicate/omitted samples, correct metric reductions, checkpoint reload at different world size where
supported, and convergence parity on a small controlled run.
""",
14: r"""
## 14.7 SFT/adapter reference

| Choice | Typical implication |
|---|---|
| Full fine-tune | Maximum flexibility; gradients/optimizer for all weights |
| LoRA | Frozen base plus small trainable low-rank updates |
| QLoRA | Quantized frozen base plus LoRA adapters |
| Attention-only targets | Small adapter; may limit behavior capacity |
| Attention + MLP | More capacity/parameters and potential regression |
| Merge adapter | Simpler single artifact; loses easy modular routing |

Fine-tuning is appropriate for stable behavior/task distribution, not rapidly changing facts. Compare
prompting and RAG first. A tiny adapter file is inseparable from its exact base model, tokenizer, and
template. Count trainable parameters and inspect matched modules. Validate that optimizer contains only
intended parameters and that ignored labels leave assistant target tokens.

Before publishing: base-versus-adapter paired evals, safety/capability retention, artifact reload in a
fresh environment, license/data documentation, merge/quantization quality checks, inference latency,
and rollback. Treat adapter inputs and outputs as model releases, not miscellaneous experiment files.
""",
15: r"""
## 15.6 Preference-optimization reference

| Method | Data/signal | Distinguishing feature |
|---|---|---|
| Reward modeling + PPO | Preferences then online reward | Explicit reward model and RL loop |
| DPO | Chosen/rejected pairs | Offline policy/reference classification objective |
| IPO/variants | Preference pairs | Alternative regularization/loss assumptions |
| KTO | Desirable/undesirable examples | Does not require paired responses |
| GRPO-style | Group rewards/verifiers | Relative group updates, common in reasoning work |

Names and implementations evolve; use current library documentation and inspect the exact loss/metrics.
Preference accuracy can improve while absolute response quality remains poor. Keep SFT-quality data or
loss, filter both-bad pairs, and evaluate objective tasks. Beta, LR, reference handling, response length,
adapter configuration, and truncation are coupled choices.

Monitor chosen/rejected log probabilities and rewards, margin/accuracy, KL/drift proxies, length,
validation loss, task metrics, safety/refusal behavior, and diversity. Audit order and style shortcuts.
Preference data encodes values and rater context; document who rated what under which guidelines.
""",
17: r"""
## 17.6 Hugging Face artifact/client reference

| Object | Purpose |
|---|---|
| `HfApi` | Repository/search/metadata operations |
| `model_info` | Card, tags, resolved SHA, provider metadata |
| `snapshot_download` | Materialize pinned repository snapshot |
| `AutoConfig` | Architecture configuration without full weights |
| `AutoTokenizer/Processor` | Text or multimodal preprocessing/template |
| `AutoModel*` | Task-specific model construction/loading |
| `InferenceClient` | Routed remote inference interface |

Cache reproducibility requires recording the resolved commit, not relying on whatever is currently in
the local cache. Use tokens through environment/secret stores and minimum permission. Accept gated
licenses deliberately. Review repository Python before remote-code execution. Model card benchmark
numbers may use different prompts, precision, templates, or contaminated data; rerun relevant evals.

On load failures check network/auth/gating, revision/file availability, disk, host RAM, device RAM,
dtype/quantization support, library architecture support, tokenizer/template, and custom-code version.
Distinguish download/load/generation time in benchmarks.
""",
}

for _number, _appendix in REFERENCE_APPENDICES.items():
    EXPANSIONS[_number].append(("md", _appendix))

REFERENCE_APPENDICES.update({
18: r"""
## 18.7 Generation reference

| Parameter | Effect | Common misuse |
|---|---|---|
| `max_new_tokens` | Hard output cap | Confusing with total context length |
| `temperature` | Rescales logits | Treating zero as universal quality setting |
| `top_k` | Keeps k candidates | Comparing k across very different distributions |
| `top_p` | Keeps cumulative mass | Assuming it guarantees factuality |
| repetition penalty | Modifies seen-token logits | Damaging code/names or required repetition |
| stop/EOS | Terminates generation | Matching strings without token-boundary care |
| seed/generator | Controls RNG stream | Claiming cross-system bitwise determinism |

Record prompt after template, model/revision, tokenizer, generation config, seed, finish reason, and
raw IDs/output. Slice generated IDs after padded input width. In batches, configure padding side and
pad token deliberately. For sampling comparisons, use repeated paired prompts and include output
length because decoding settings change how long the model speaks.

Structured output guarantees at most syntactic/schema compliance supported by the engine. Validate
business semantics and unknown fields, define retry/repair policy, and treat parser failure as an
observed outcome. Streaming must accumulate before final structured parsing.
""",
19: r"""
## 19.6 Semantic-search reference

| Stage | Primary metric/question |
|---|---|
| Encoder selection | Does its training recipe match query/document domain? |
| Exact baseline | What is best achievable recall for these embeddings? |
| ANN index | Recall loss versus latency/memory/build/update cost |
| Metadata filtering | Are ACL/tenant/time constraints correct? |
| Hybrid retrieval | Does lexical+dense improve slices? |
| Reranking | Does precision rise without losing required recall/latency? |

Cosine equals dot product only after unit normalization. Euclidean ranking of normalized vectors is
monotonically related, but index configuration must match. Similarity magnitudes are not calibrated
relevance probabilities and shift by model/domain/query length. Tune thresholds on labeled data and
include a no-result option.

Version embedding model/revision, preprocessing/prefix, dimension, normalization, distance function,
index parameters, corpus snapshot, and IDs. Never mix vector generations. Evaluate queries with
multiple relevant documents and incomplete judgments carefully; inspect failures qualitatively.
""",
20: r"""
## 20.8 RAG reference architecture

`sources → parse → normalize → chunk → metadata/ACL → embed → index/version`

`question+history → standalone query → filters → lexical/dense retrieve → fuse/rerank → deduplicate
→ token budget → prompt with source IDs → generate → validate citations/claims → answer or abstain`

Every arrow is a testable boundary. Preserve IDs and versions through the trace. Gold-support retrieval
and gold-context generation experiments separate stage ceilings. Context relevance is not faithfulness;
faithfulness is not answer correctness; formatted citations are not entailed citations.

Common failures: header/footer chunks dominate; overlap duplicates crowd top-k; wrong tenant filters;
stale/deleted sources; query rewrite changes intent; approximate index loses rare support; reranker
prefers stylistic overlap; token budget drops the decisive chunk; conflicting sources are blended;
model follows injected document instructions; citations reference unsupported or nonexistent IDs.

Production ownership includes source freshness/deletion, index migrations, ACL audits, evaluation,
security response, and observable stage latency—not only the generation prompt.
""",
22: r"""
## 22.6 Agent/tool reference

| Layer | Responsibility |
|---|---|
| Model | Propose calls/arguments and synthesize result |
| Loop/host | Maintain state, budgets, call/result correlation |
| Validator | Enforce JSON types and semantic constraints |
| Authorizer | Bind user identity, ownership, permission, approval |
| Tool service | Execute idempotently with timeout and scoped credentials |
| Evaluator/trace | Verify trajectory, effects, answer, cost, and safety |

Stopping conditions include final response, step/tool/token/deadline budget, repeated calls, nonretryable
error, cancellation, or approval denial. Parallelize only independent calls. Treat all tool results as
untrusted bounded data. Tool success does not mean task success; final eloquence does not mean calls
were safe/correct.

Evaluate tool selection, argument exactness, unnecessary calls, ordering/dependencies, error recovery,
side effects, citation/use of observations, final correctness, latency, and cost. Run deterministic
workflows without an agent when the state graph is known—the simplest sufficient controller is usually
more reliable and auditable.
""",
24: r"""
## 24.6 MCP reference

| Primitive | Meaning |
|---|---|
| Tool | Callable operation with input schema |
| Resource | URI-addressed readable content/context |
| Resource template | Parameterized resource URI pattern |
| Prompt | Discoverable reusable prompt template |
| Sampling | Server request for host-mediated model generation |
| Capability | Negotiated feature support during initialization |
| Transport | Message channel such as stdio or HTTP-based remote transport |

MCP standardizes exchange, not trust. The host remains responsible for server identity, installation/
connection approval, user authorization, tool policy, input validation, result size, untrusted content,
credential scoping, and audit. Stdio processes can access what their OS identity can access. Remote
servers add authentication, TLS/origin/redirect, tenancy, and availability concerns.

Test domain functions, server protocol discovery/invocation, host schema conversion, model behavior,
and security policy separately. Pin protocol/SDK/server versions and test compatibility. Avoid broad
filesystem/shell/browser tools unless strongly isolated and explicitly authorized.
""",
28: r"""
## 28.8 Evaluation reference

| Evaluation layer | Examples |
|---|---|
| Input/data | schema, distribution, leakage, slice coverage |
| Component | retrieval recall, tool args, parser validity |
| Output | correctness, constraints, citations, style |
| Trajectory/effects | calls, authorization, side effects, step budget |
| Operational | TTFT/latency, tokens, cost, error/fallback |
| Human/product | preference, task completion, escalation, satisfaction |

Always state numerator, denominator, unit of sampling, aggregation, and uncertainty. Pair comparisons
on the same examples. Separate critical release gates from optimization metrics. Store raw outputs and
grader details. Treat parse/timeouts/refusals as outcomes according to predeclared policy, not missing
data deleted after the fact.

Evaluation is iterative but protect a final test from overfitting. Add production failures to regression
coverage while refreshing broad representative samples. A metric becomes unreliable when optimized
without checking its relationship to user value—keep qualitative review and multiple independent
signals.
""",
29: r"""
## 29.7 LLM-judge reference

| Risk | Control/evidence |
|---|---|
| Position bias | Randomize and swap A/B; report inconsistency |
| Verbosity/style bias | Explicit rubric, length slices, decisive evidence |
| Self/family preference | Blind identities; compare judge families/humans |
| Prompt injection | Candidate as untrusted data; adversarial calibration |
| Bad reference | Reference audit/multiple references/expert review |
| Nondeterminism | Repeats/seeds/order; raw judgments |
| Drift | Pin/version judge and recalibrate after changes |

Validate structured output before scoring. Preserve parse/call failures. Calibrate on representative
human labels and report by criterion/slice. Pair judge decisions with deterministic graders whenever
the requirement is executable. Use thresholds only after examining calibration and uncertainty.

A judge can support ranking, triage, and scalable qualitative measurement. It should not be the sole
authority for high-stakes facts, safety, security, or consequential actions. Route disagreements,
critical errors, novel slices, and random samples to humans.
""",
30: r"""
## 30.8 Security-control reference

| Security property | Enforced by |
|---|---|
| Authentication | Verified user/service identity outside model |
| Authorization | Server-side policy bound to identity/resource/action |
| Input safety | Parser/schema/semantic validation and resource limits |
| Isolation | Process/container/network/filesystem/tenant boundaries |
| Side-effect safety | Idempotency, exact approval, transactions/reversibility |
| Confidentiality | Least data, scoped credentials, encryption/redaction/retention |
| Availability | Budgets, bounded queues, rate limits, timeouts, circuit isolation |
| Auditability | Tamper-resistant effect/decision logs with privacy controls |

Prompts and model refusals are behavioral controls, not security enforcement. Assume untrusted users,
documents, images, tool results, and MCP servers can fully control model output. Design so compromised
output still cannot cross deterministic authorization/isolation boundaries.

Measure security using actual traces/effects and invariants. Maintain revocation/kill switches and an
incident plan. Re-threat-model whenever tools, data sources, tenancy, model, deployment, or persistent
memory changes.
""",
32: r"""
## 32.8 Multimodal reference

| Task | Prefer/evaluate |
|---|---|
| Embedded PDF text | Native extraction plus rendered verification |
| Scanned text | OCR CER/WER and field accuracy |
| Forms/invoices | Structured schema, arithmetic/business validation, evidence region |
| Tables | Cell/structure accuracy, headers, merged cells |
| Charts | Axis/legend perception plus numeric reasoning tolerance |
| General images | Grounded descriptions, object/spatial slices |

Record source hash, page/image index, original/transformed dimensions, processor/model revision, crop/
tile coordinates, prompt, schema, and output. Evaluate illegible/missing-content abstention. Batch by
visual token load, not image count alone.

Files are untrusted complex inputs. Verify type/signature, sandbox parsers, cap bytes/pixels/pages/time/
decompression, enforce tenant ACL, and treat visible/hidden instructions as injection. Protect biometric,
private, copyrighted, and location-sensitive content according to policy. Validate high-value numeric
fields outside the model.
""",
33: r"""
## 33.8 Production-readiness reference

| Area | Minimum evidence |
|---|---|
| Quality | Frozen evals, critical slices, canary comparison |
| Latency | Stage spans and p50/p95/p99 under representative load |
| Capacity | Sustainable tokens/sec, queues, headroom, autoscaling lag |
| Reliability | Deadlines, bounded retries/queues, cancellation, fallback tests |
| Observability | Metrics/logs/traces with versions and privacy policy |
| Security | Authn/z, limits, secrets, isolation, incident controls |
| Release | Immutable artifacts, staged rollout, rollback exercise |

An SLO should count only valid useful responses as success. Track error-budget burn across dependency,
validation, timeout, and quality failures. Avoid high-cardinality metrics and sensitive prompt labels.
Propagate request/trace IDs through retrieval, tools, model, and validation.

Load tests must avoid coordinated omission, include prompt/output distributions and cancellations, and
run long enough for queues/autoscaling. Capacity conclusions are invalid without hardware/software and
quality/length controls.
""",
36: r"""
## 36.8 Ollama operations reference

| Concern | Minimum practice |
|---|---|
| Reproducibility | Pin/record model digest, tag, quantization, Modelfile, Ollama version |
| API | Test a narrow native or compatible endpoint contract |
| Capacity | Measure cold load, prefill, decode, concurrency, memory, and context lengths |
| Structure | Constrain with schema, then validate semantics and business rules |
| Embeddings | Pin model/dimension/preprocessing; reject unexpected truncation |
| Security | Loopback by default; gateway auth/TLS/limits for intentional exposure |
| Lifecycle | Inventory models; stage updates; retain rollback and deletion policy |

Useful commands include `ollama pull`, `ollama list`, `ollama ps`, `ollama show`, `ollama run`, and
`ollama create`. Run lifecycle mutations deliberately in a terminal, not as hidden notebook side
effects. A Modelfile captures a base, parameters, template, system behavior, adapters, and license, but
does not replace an application configuration or model card.

Treat generated JSON, tool calls, paths, and URLs as untrusted. Bound request and response tokens,
timeouts, queues, concurrency, and tool effects. Redact sensitive telemetry and test behavior after any
model, template, quantization, server, or hardware change.
""",
37: r"""
## 37.10 vLLM operations reference

| Question | Evidence to collect |
|---|---|
| Does it fit? | Weights + total concurrent KV + workspace/fragmentation margin |
| Is it fast? | TTFT, ITL, end-to-end percentiles, tokens/sec under realistic load |
| Does it preserve quality? | Pinned-model eval after dtype/quantization/kernel/config changes |
| Does it scale? | Per-topology throughput, communication, queueing, failure behavior |
| Is it compatible? | Chat template, tools/reasoning parser, structured schema, modalities |
| Is it operable? | health/readiness, metrics, drains, canary/rollback, OOM/failure tests |

Pin vLLM/image digest, model and tokenizer commits, template, serve config, driver/CUDA, and hardware.
Use current CLI/API docs because flags and structured/tool behavior evolve. OpenAI-compatible means a
protocol surface, not identical semantics/features.

Protect ingress with authenticated TLS, per-tenant token/concurrency limits, payload bounds, and private
admin endpoints. Control remote code and adapter loading. Monitor total cached tokens/KV pressure,
queue time, prefix hit rate, aborts/errors, and GPU health—not GPU utilization alone.
""",
})

for _number in [14, 15, 17, 18, 19, 20, 22, 24, 28, 29, 30, 32, 33, 36, 37]:
    EXPANSIONS[_number].append(("md", REFERENCE_APPENDICES[_number]))

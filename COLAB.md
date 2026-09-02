# Running the Course in Google Colab

Each notebook is standalone: upload it to Colab or open it from a GitHub repository, then
run the setup cell directly below the title. You do not need to clone the project or run
`uv sync` inside Colab.

## 1. Select a runtime

Use **Runtime → Change runtime type**.

| Notebook type | Recommended runtime |
|---|---|
| Mathematics and evaluation mechanics | CPU is sufficient |
| Small local model inference | T4 GPU or better |
| LoRA/DPO training | T4 for the 0.5B examples; L4/A100 is faster |
| QLoRA, longer sequences, larger models | L4/A100 with High-RAM when available |
| vLLM server | Use a dedicated supported Linux accelerator host; Colab is best used as its client |

Colab hardware is not guaranteed. Always inspect the accelerator printed by the setup cell.

## 2. Configure Hugging Face access

Open the key icon in the left sidebar (**Secrets**) and add:

- Name: `HF_TOKEN`
- Value: your Hugging Face token
- Enable **Notebook access**

Use a token with the minimum permissions needed. Read access is enough for public/gated
models after accepting their licenses; Inference Providers calls require the corresponding
account permission. The setup cell copies the secret into the runtime environment but never
prints it or stores it in the notebook.

### VS Code Colab extension

The VS Code extension still executes Python inside the remote Colab VM. It cannot read the `.env`
file on your laptop unless that file is explicitly uploaded—which you should not do for credentials.
Add `HF_TOKEN` through Colab's browser Secrets panel and enable notebook access, then reconnect or
rerun the setup cell from VS Code. The course detects Colab by the installed `google.colab` module
and Colab runtime variables, even when the extension attaches before that module has been imported.

To validate authentication without displaying the token:

```python
import os
from huggingface_hub import HfApi

token = os.environ.get("HF_TOKEN")
assert token, "HF_TOKEN is not available in this runtime"
identity = HfApi(token=token).whoami()
print("Connected to Hugging Face as:", identity["name"])
```

If presence is still reported as false, open the same notebook once in the Colab browser, confirm
the secret name is exactly `HF_TOKEN`, turn on its notebook-access toggle, and reconnect the runtime.
Secrets and installed packages disappear from process state when the remote runtime is replaced, so
rerun the setup cell after every reconnect.

## 3. Run the setup cell

The tagged `setup` / `colab` cell:

1. Detects Colab.
2. Installs only that notebook's extra dependencies.
3. Loads `HF_TOKEN` or `HUGGINGFACE_TOKEN` from Secrets.
4. Reports Python, operating system, and accelerator.
5. Warns when a training notebook has no CUDA GPU.

Colab already includes PyTorch, so the setup cell does not replace it. This avoids large
downloads and CUDA-wheel mismatches.

If Colab asks for a runtime restart after a package change, restart and rerun the setup cell.

## 4. Enable expensive cells deliberately

Training notebooks use:

```python
RUN_TRAINING = False
```

Read and adjust the configuration first, confirm a CUDA GPU is active, then change it to
`True`. Remote inference cells check for a token and skip safely when it is unavailable.

Useful checks:

```python
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
print(torch.cuda.get_device_properties(0).total_memory / 2**30 if torch.cuda.is_available() else 0)
```

## 5. Preserve checkpoints across runtime resets

Colab's local filesystem is temporary. Mount Drive only when you need persistence:

```python
from google.colab import drive
drive.mount("/content/drive")
OUTPUT_DIR = "/content/drive/MyDrive/llm-course/checkpoints/my-run"
```

Then set the trainer's `output_dir=OUTPUT_DIR`. Avoid saving every few steps to Drive; many
small synchronous writes slow training. Save periodic checkpoints and keep `save_total_limit`
small. Do not store `.env` files or access tokens in Drive-backed output directories.

For larger runs, push adapters to a private Hugging Face repository or copy final artifacts
to Drive after training.

## 6. Recover from out-of-memory errors

Change one variable at a time:

1. Reduce `per_device_train_batch_size`.
2. Increase `gradient_accumulation_steps` to preserve the effective batch.
3. Reduce `max_length` after verifying important tokens are not truncated.
4. Enable gradient checkpointing.
5. Use LoRA/QLoRA and an appropriate low-bit base model.
6. Reduce LoRA target modules or rank.
7. Restart the runtime after failed runs if memory remains fragmented.

Record the actual GPU, package versions, model revision, dataset revision, precision, and
effective global batch with every experiment.

## 7. Colab limitations

- Runtime storage and processes disappear when the session resets.
- GPU models and availability vary.
- Long-running servers may be disconnected.
- Multi-GPU and serious vLLM serving require dedicated infrastructure.
- Installing CUDA extensions such as FlashAttention can take a long time or fail when the
  runtime compiler/CUDA/PyTorch versions do not match. The course uses PyTorch's scaled-dot-
  product attention demonstration instead of compiling `flash-attn` in Colab.

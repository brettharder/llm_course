#!/usr/bin/env python3
"""Static validation for generated course notebooks."""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path

ROOT = Path(__file__).parent
paths = sorted(ROOT.glob("[0-9][0-9]_*/*.ipynb"))
errors: list[str] = []
numbers: list[int] = []

for path in paths:
    try:
        notebook = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        errors.append(f"{path}: invalid JSON: {exc}")
        continue

    if notebook.get("nbformat") != 4:
        errors.append(f"{path}: expected nbformat 4")
    number = int(path.stem.split("_", 1)[0])
    numbers.append(number)
    title = notebook.get("cells", [{}])[0].get("source", "")
    if not re.search(rf"# Notebook {number:02d}\b", title):
        errors.append(f"{path}: first cell title does not match notebook number")
    cells = notebook.get("cells", [])
    if len(cells) < 2 or cells[1].get("cell_type") != "code":
        errors.append(f"{path}: missing Colab setup as second cell")
    else:
        setup_tags = set(cells[1].get("metadata", {}).get("tags", []))
        setup_source = cells[1].get("source", "")
        if not {"setup", "colab"}.issubset(setup_tags) or "google.colab" not in setup_source:
            errors.append(f"{path}: invalid Colab setup cell metadata/source")

    ids: set[str] = set()
    markdown_words = 0
    lesson_code_cells = 0
    for index, cell in enumerate(notebook.get("cells", []), 1):
        cell_id = cell.get("id")
        if not cell_id or cell_id in ids:
            errors.append(f"{path}: missing/duplicate id at cell {index}")
        ids.add(cell_id)
        if cell.get("cell_type") == "code":
            if "setup" not in cell.get("metadata", {}).get("tags", []):
                lesson_code_cells += 1
            if cell.get("outputs"):
                errors.append(f"{path}: saved output in cell {index}")
            if cell.get("execution_count") is not None:
                errors.append(f"{path}: execution count in cell {index}")
            try:
                ast.parse(cell.get("source", ""), filename=f"{path}:cell-{index}")
            except SyntaxError as exc:
                errors.append(f"{path}: cell {index} syntax error: {exc}")
        elif cell.get("cell_type") == "markdown":
            markdown_words += len(re.findall(r"\b[\w'-]+\b", cell.get("source", "")))

    if markdown_words < 700:
        errors.append(f"{path}: lesson depth regression ({markdown_words} markdown words; minimum 700)")
    if lesson_code_cells < 3:
        errors.append(f"{path}: needs at least 3 non-setup code examples")
    if len(notebook.get("cells", [])) < 12:
        errors.append(f"{path}: needs at least 12 lesson/setup cells")

    serialized = json.dumps(notebook).lower()
    for forbidden in ("import anthropic", "from openai import", "openai_api_key", "anthropic_api_key"):
        if forbidden in serialized:
            errors.append(f"{path}: forbidden proprietary SDK/key reference: {forbidden}")
    if re.search(r"hf_[a-z0-9]{20,}", serialized):
        errors.append(f"{path}: possible embedded Hugging Face token")

expected = list(range(1, len(paths) + 1))
if numbers != expected:
    errors.append(f"notebook numbering is not contiguous: {numbers}")

if errors:
    print("VALIDATION FAILED")
    for error in errors:
        print(" -", error)
    raise SystemExit(1)

print(f"Validated {len(paths)} comprehensive notebooks; depth, Colab setup, JSON, cell syntax, outputs, and provider scan passed.")

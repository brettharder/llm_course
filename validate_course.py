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


def source_text(cell: dict) -> str:
    """Return source for either valid nbformat representation."""
    source = cell.get("source", "")
    return "".join(source) if isinstance(source, list) else source


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
    title = source_text(notebook.get("cells", [{}])[0])
    if not re.search(rf"# Notebook {number:02d}\b", title):
        errors.append(f"{path}: first cell title does not match notebook number")
    objective_count = len(re.findall(r"^\s*- ", title, flags=re.MULTILINE))
    if objective_count < 3:
        errors.append(f"{path}: introduction needs at least 3 explicit learning objectives")
    cells = notebook.get("cells", [])
    if len(cells) < 2 or cells[1].get("cell_type") != "code":
        errors.append(f"{path}: missing Colab setup as second cell")
    else:
        setup_tags = set(cells[1].get("metadata", {}).get("tags", []))
        setup_source = source_text(cells[1])
        if not {"setup", "colab"}.issubset(setup_tags) or "google.colab" not in setup_source:
            errors.append(f"{path}: invalid Colab setup cell metadata/source")
        if 'userdata.get("HF_TOKEN")' not in setup_source:
            errors.append(f"{path}: setup must read HF_TOKEN from the Colab Secrets UI")
        if "getpass" in setup_source:
            errors.append(f"{path}: setup must not request credentials inside notebook cells")

    ids: set[str] = set()
    markdown_words = 0
    lesson_code_cells = 0
    lesson_headings = 0
    numbered_sections: list[str] = []
    primary_reference_links = 0
    for index, cell in enumerate(notebook.get("cells", []), 1):
        source = source_text(cell)
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
                ast.parse(source, filename=f"{path}:cell-{index}")
            except SyntaxError as exc:
                errors.append(f"{path}: cell {index} syntax error: {exc}")
        elif cell.get("cell_type") == "markdown":
            markdown_words += len(re.findall(r"\b[\w'-]+\b", source))
            lesson_headings += len(re.findall(r"^##+ ", source, flags=re.MULTILINE))
            numbered_sections.extend(re.findall(r"^##+ (\d+\.\d+)\b", source, flags=re.MULTILINE))
            if "## Primary references and further study" in source:
                primary_reference_links += len(re.findall(r"\[[^]]+\]\(https?://[^)]+\)", source))

    if markdown_words < 840:
        errors.append(f"{path}: lesson depth regression ({markdown_words} markdown words; minimum 840)")
    if lesson_code_cells < 5:
        errors.append(f"{path}: needs at least 5 non-setup code examples")
    if lesson_headings < 7:
        errors.append(f"{path}: needs at least 7 substantive lesson headings")
    if len(notebook.get("cells", [])) < 16:
        errors.append(f"{path}: needs at least 16 lesson/setup cells")
    wrong_prefixes = [section for section in numbered_sections if int(section.split(".", 1)[0]) != number]
    if wrong_prefixes:
        errors.append(f"{path}: section numbers do not match notebook number: {wrong_prefixes}")
    duplicate_sections = sorted({section for section in numbered_sections if numbered_sections.count(section) > 1})
    if duplicate_sections:
        errors.append(f"{path}: duplicate numbered sections: {duplicate_sections}")
    if primary_reference_links < 2:
        errors.append(f"{path}: needs at least 2 curated primary/official reference links")
    exercise_source = next(
        (source_text(cell) for cell in reversed(cells) if cell.get("cell_type") == "markdown" and "## Exercises" in source_text(cell)),
        "",
    )
    exercise_count = len(re.findall(r"^\s*\d+\. ", exercise_source, flags=re.MULTILINE))
    if exercise_count < 3 or "## Checkpoint" not in exercise_source:
        errors.append(f"{path}: needs at least 3 numbered exercises and a checkpoint prompt")

    serialized = json.dumps(notebook).lower()
    for forbidden in ("import anthropic", "from openai import", "openai_api_key", "anthropic_api_key"):
        if forbidden in serialized:
            errors.append(f"{path}: forbidden proprietary SDK/key reference: {forbidden}")
    if re.search(r"hf_[a-z0-9]{20,}", serialized):
        errors.append(f"{path}: possible embedded Hugging Face token")
    if re.search(r"/(users|home)/[^/]+/", serialized):
        errors.append(f"{path}: machine-specific absolute path is not Colab portable")

expected = list(range(1, len(paths) + 1))
if numbers != expected:
    errors.append(f"notebook numbering is not contiguous: {numbers}")

if errors:
    print("VALIDATION FAILED")
    for error in errors:
        print(" -", error)
    raise SystemExit(1)

print(f"Validated {len(paths)} comprehensive notebooks; rigorous depth, examples, structure, Colab setup, JSON, cell syntax, outputs, and provider scan passed.")

# Original Project Modification Notes

This document marks the locations changed from the original `VLM Weather Benchmark` project so the open-source project matches the MetServ-VL paper.

## 1. Project Identity And Paper Alignment

Changed files:

- `README.md`
- `LICENSE`
- `pyproject.toml`
- `src/vlm_weather/__init__.py`

Purpose:

- Rename the public-facing project from the generic `VLM Weather Benchmark` wording to `MetServ-VL`.
- Update metadata, license holder, authors, repository URL, citation, and paper result tables.
- Keep the Python import package as `vlm_weather` to avoid breaking the original engineering structure.

Paper mapping:

- Title: `Vision-Language Models for Meteorological Services: A High-Quality Data-Centric Approach`.
- Contribution 3: code and benchmark release for expert-annotated meteorological service evaluation.

## 2. LLM-As-Judge Protocol

Changed file:

- `src/vlm_weather/evaluators/llm_judge.py`

Purpose:

- Replace the placeholder prompt with the paper's Fig. 8 evaluation protocol.
- Add the background region knowledge base for 江南, 华南, 淮南, 东北, 江汉, 江淮, 黄淮, 华北, 新疆伊犁河谷, and 内蒙古河套地区.
- Implement the five scoring dimensions and weights:
  - Time Accuracy: 15%
  - Region Accuracy: 25%
  - Weather Phenomena Accuracy: 25%
  - Completeness: 20%
  - No Hallucination: 15%
- Recompute `total_score` in code to ensure consistency with the weighted formula.
- Strip `<think>...</think>` blocks and parse strict JSON robustly for reasoning-style models.

Paper mapping:

- Section V-B/C: LLM-based judgment.
- Fig. 8: prompt and evaluation protocol for LLM-as-Judge.
- Table II: LLM-as-Judge evaluation results.

## 3. Data Construction Utility Fix

Changed file:

- `src/vlm_weather/common.py`

Purpose:

- Add the missing `file_stem()` helper used by:
  - `src/vlm_weather/extractors/docx.py`
  - `src/vlm_weather/extractors/docx_images.py`
- Without this helper, DOCX extraction modules fail during import.

Paper mapping:

- Section III-C: multi-stage data construction pipeline.
- Section III-D: staged training data preparation from image-text pairs.

## 4. Dataset Documentation

Changed file:

- `docs/DATASET.md`

Purpose:

- Separate the paper's full data assets from the local public repository subset.
- Document `Dexpert`, `Daug`, and `Db`.
- Record the local data statistics:
  - 109 images in `data/dataset/images/`
  - 470 entries in `data/dataset/labels/vlm_test_dataset.json`
  - 109 entries in `data/dataset/labels/vlm_test_dataset1.json`
- Explain the unified evaluation JSON format used by the paper.

Paper mapping:

- Section III-A: multimodal data assets.
- Section V-B: benchmark evaluation setting and sample format.

## 5. README Reproduction Entry Points

Changed file:

- `README.md`

Purpose:

- Add commands for lexical metrics, embedding similarity, Qwen3 embedding similarity, and LLM-as-Judge.
- Add the MetServ-VL paper's quantitative result tables.
- Clarify that this repository releases the benchmark subset and tooling, while full private operational data and training weights are not redistributed here.

Paper mapping:

- Section V-A: experiment setup.
- Section V-C: quantitative evaluation and LLM-as-Judge.

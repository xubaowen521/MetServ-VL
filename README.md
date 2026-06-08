# MetServ-VL

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> Open-source benchmark and evaluation toolkit for **Vision-Language Models for Meteorological Services: A High-Quality Data-Centric Approach**.

English | [中文](#中文说明)

## Overview

MetServ-VL is a data-centric framework for adapting vision-language models to meteorological service scenarios. The paper builds a scalable multimodal data construction pipeline, a curriculum learning training strategy, and an expert-annotated benchmark for Chinese meteorological services.

This repository contains the open benchmark subset and reproducible tooling:

- **Benchmark data**: meteorological service image samples and expert/model answer JSON files.
- **Evaluation scripts**: comprehensive automatic and expert-aligned evaluation protocols for assessing generation quality, semantic consistency, and meteorological factual correctness.
- **Data construction utilities**: DOCX/PDF extraction helpers for building image-text pairs from operational meteorological documents.

The package import path is kept as `vlm_weather` for compatibility with the original engineering skeleton.

## Repository Contents

```
MetServ-VL/
├── data/
│   └── dataset/
│       ├── images/                 # Public benchmark image subset
│       ├── labels/
│       │   ├── vlm_test_dataset.json
│       │   └── vlm_test_dataset1.json
│       └── image-caption.json
├── docs/
│   ├── DATASET.md                  # Benchmark format and statistics
│   └── MODIFICATION_NOTES.md       # Locations changed from the original project
├── examples/
│   └── run_evaluation.py
├── src/
│   └── vlm_weather/
│       ├── common.py
│       ├── extractors/             # Data construction utilities
│       └── evaluators/             # Evaluation protocols
├── pyproject.toml
├── requirements.txt
└── README.md
```

## Installation

```bash
pip install -r requirements.txt
pip install -e .
```

Optional environment variables:

| Variable | Description | Default |
| --- | --- | --- |
| `BERT_MODEL_PATH` | BERT model used by BERTScore | `bert-base-chinese` |
| `EMBEDDING_MODEL_PATH` | SentenceTransformer model | `Qwen/Qwen3-Embedding-4B` |
| `QWEN3_EMBEDDING_MODEL` | Qwen3 embedding model | `Qwen/Qwen3-Embedding-0.6B` |
| `LLM_BASE_URL` | OpenAI-compatible LLM endpoint | unset |
| `LLM_API_KEY` | LLM judge API key | `EMPTY` |
| `LLM_MODEL` | LLM judge model name | `deepseekr1` |

## Data Format

Evaluation files use the unified format described in the paper:

```json
{
  "index": 0,
  "image_path": "200000694-图1.jpg",
  "question": "请基于提供的预报图给出具体预报",
  "human_answer": "未来10天（18-27日），江南西南部、华南北部...",
  "assistant_answer": "未来10天（18-27日），新疆北部、西北地区东南部..."
}
```



## Evaluation

### Lexical And Traditional NLG Metrics

```bash
python -m vlm_weather.evaluators.compute_score \
    --file data/dataset/labels/vlm_test_dataset.json \
    --output_path data/results \
    --bert_model bert-base-chinese
```

Metrics: BLEU-1/4/L, ROUGE-1/2/L, METEOR, CIDEr, and BERTScore.

### SentenceTransformer Similarity

```bash
python -m vlm_weather.evaluators.embed_score \
    --file data/dataset/labels/vlm_test_dataset.json \
    --output data/results/similarity_results.json \
    --model_path Qwen/Qwen3-Embedding-4B
```

### Qwen3-Embedding Similarity

```bash
python -m vlm_weather.evaluators.qwen3_embed \
    --file data/dataset/labels/vlm_test_dataset.json \
    --output_path data/results \
    --model Qwen/Qwen3-Embedding-0.6B
```

### LLM-As-Judge

The LLM-as-Judge implementation follows the paper's five-dimensional protocol:

| Dimension | Weight | Focus |
| --- | ---: | --- |
| Time Accuracy | 15% | Date/time range consistency |
| Region Accuracy | 25% | Correct region naming and macro-region mapping |
| Weather Phenomena Accuracy | 25% | Rainfall levels, wind strength, and weather type |
| Completeness | 20% | Coverage of key events and locations |
| No Hallucination | 15% | Avoiding invented regions, events, or phenomena |

```bash
python -m vlm_weather.evaluators.llm_judge \
    --input data/dataset/labels/vlm_test_dataset.json \
    --output data/results/vlm_test_dataset_scored.json \
    --base_url http://your-llm-host:5090/v1 \
    --api_key your-key \
    --model deepseekr1
```

## Data Construction Utilities

```bash
# Extract text and inline images from DOCX files
python -m vlm_weather.extractors.docx \
    --input_dir /path/to/docx/files \
    --output_dir /path/to/output \
    --start_id 800002069

# Extract images through DOCX relationships
python -m vlm_weather.extractors.docx_images extract \
    --input_dir /path/to/docx/files \
    --output_dir /path/to/images \
    --min_height 500

# Extract plain text from a PDF
python -m vlm_weather.extractors.pdf \
    --input /path/to/file.pdf \
    --output /path/to/output.txt
```

## Paper Results

The paper reports the following benchmark results for MetServ-VL:

| Model | BLEU-1/4 | BLEU-L | ROUGE-1/2 | ROUGE-L | METEOR | BERTScore | Sim(BGEM3) |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: |
| MetServ-VL(SFT1) | 69.59/46.02 | 36.52 | 78.34/58.12 | 69.30 | 0.6601 | 0.9289 | 0.9202 |
| MetServ-VL(SFT2) | 76.52/50.98 | 39.86 | 84.93/63.81 | 71.85 | 0.7327 | 0.9341 | 0.9312 |
| MetServ-VL | 81.25/53.75 | 40.80 | 87.24/68.08 | 73.87 | 0.7617 | 0.9526 | 0.9436 |

LLM-as-Judge scores reported in the paper:

| Agent Type | Method | Score (%) |
| --- | --- | ---: |
| Single Agent | DeepSeek-R1 | 78.34 |
| Single Agent | Qwen3-32B | 84.68 |
| Single Agent | GLM4-230B | 85.54 |
| Single Agent | Qwen3-30B-A3B | 82.76 |
| Multi Agent | Majority Voting | 79.07 |
| Multi Agent | Weighted Averaging | 82.83 |



## 中文说明

MetServ-VL 是面向气象服务场景的视觉语言模型数据中心方法。本文围绕高质量气象多模态数据构建、课程学习训练策略和中国区域专家标注评测基准展开。

本仓库提供论文开源内容中的评测集子集、评测脚本和数据构建工具：

- `data/dataset/images/`：公开气象服务图像样例。
- `data/dataset/labels/`：专家参考答案和模型生成答案。
- `src/vlm_weather/evaluators/`：自动指标、语义相似度和 LLM-as-Judge 评测。
- `src/vlm_weather/extractors/`：从 DOCX/PDF 构建图文配对数据的工具。
- `docs/MODIFICATION_NOTES.md`：相对原始工程的修改位置标注。

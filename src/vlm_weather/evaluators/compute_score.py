"""Traditional NLG evaluation metrics: ROUGE, BLEU, METEOR, CIDEr, BERTScore."""
from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any

import jieba
import nltk
import numpy as np
import torch
from bert_score import score as bert_score_fn
from nltk.translate.meteor_score import meteor_score
from pycocoevalcap.cider.cider import Cider
from rouge import Rouge
from sacrebleu.metrics import BLEU

logger = logging.getLogger(__name__)

DEFAULT_BERT_MODEL = os.environ.get("BERT_MODEL_PATH", "bert-base-chinese")
DEFAULT_NLTK_DIR = os.environ.get(
    "NLTK_DATA_DIR",
    str(Path(__file__).with_name("nltk_data")),
)
Path(DEFAULT_NLTK_DIR).mkdir(parents=True, exist_ok=True)
nltk.data.path.append(DEFAULT_NLTK_DIR)
nltk.download("wordnet", download_dir=DEFAULT_NLTK_DIR, quiet=True)
nltk.download("punkt", download_dir=DEFAULT_NLTK_DIR, quiet=True)

_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
logger.info("Using device: %s", _DEVICE)


def tokenize_chinese(text: str) -> str:
    """Tokenize Chinese text using jieba."""
    return " ".join(jieba.lcut(text))


def calculate_bert_score(
    references: list[str],
    candidates: list[str],
    model_type: str = DEFAULT_BERT_MODEL,
) -> float | None:
    """Compute BERTScore F1 using a Chinese pre-trained model.

    Args:
        references: List of reference texts.
        candidates: List of candidate texts.
        model_type: BERT model identifier or path.

    Returns:
        Average F1 score, or ``None`` on error.
    """
    try:
        _p, _r, f1 = bert_score_fn(
            candidates,
            references,
            model_type=model_type,
            num_layers=12,
            lang="zh",
            verbose=False,
            device=_DEVICE,
        )
        avg_f1 = float(f1.mean())
        logger.info("BERTScore (F1): %.4f", avg_f1)
        return avg_f1
    except Exception as exc:
        logger.error("BERTScore computation failed: %s", exc)
        return None


def calculate_bleu_scores(
    references: list[str],
    candidates: list[str],
) -> tuple[float | None, float | None, float | None]:
    """Compute BLEU-1, BLEU-4, and BLEU-L (n=6) scores.

    Returns:
        Tuple of (bleu1, bleu4, bleul), or ``None`` for each on error.
    """
    try:
        bleu1 = BLEU(max_ngram_order=1).corpus_score(candidates, [references]).score
        bleu4 = BLEU(max_ngram_order=4).corpus_score(candidates, [references]).score
        bleul = BLEU(max_ngram_order=6).corpus_score(candidates, [references]).score
        logger.info("BLEU-1: %.2f | BLEU-4: %.2f | BLEU-L: %.2f", bleu1, bleu4, bleul)
        return bleu1, bleu4, bleul
    except Exception as exc:
        logger.error("BLEU computation failed: %s", exc)
        return None, None, None


def calculate_rouge_score(
    references: list[str],
    candidates: list[str],
) -> tuple[float | None, float | None, float | None]:
    """Compute ROUGE-1, ROUGE-2, and ROUGE-L F1 scores.

    Returns:
        Tuple of (rouge_1, rouge_2, rouge_l), or ``None`` for each on error.
    """
    try:
        scores = Rouge().get_scores(candidates, references, avg=True)
        r1 = scores["rouge-1"]["f"]
        r2 = scores["rouge-2"]["f"]
        rl = scores["rouge-l"]["f"]
        logger.info("ROUGE-1: %.4f | ROUGE-2: %.4f | ROUGE-L: %.4f", r1, r2, rl)
        return r1, r2, rl
    except Exception as exc:
        logger.error("ROUGE computation failed: %s", exc)
        return None, None, None


def calculate_meteor_score(
    references: list[str],
    candidates: list[str],
) -> float | None:
    """Compute METEOR score (token-level average).

    Returns:
        Average METEOR score, or ``None`` on error.
    """
    try:
        scores = [
            meteor_score([ref.split()], cand.split())
            for ref, cand in zip(references, candidates)
        ]
        avg = float(np.mean(scores))
        logger.info("METEOR: %.4f", avg)
        return avg
    except Exception as exc:
        logger.error("METEOR computation failed: %s", exc)
        return None


def calculate_cider_score(
    references: list[str],
    candidates: list[str],
) -> float | None:
    """Compute CIDEr score.

    Returns:
        CIDEr score, or ``None`` on error.
    """
    try:
        refs = {i: [ref] for i, ref in enumerate(references)}
        cands = {i: [cand] for i, cand in enumerate(candidates)}
        score, _ = Cider().compute_score(refs, cands)
        logger.info("CIDEr: %.4f", score)
        return float(score)
    except Exception as exc:
        logger.error("CIDEr computation failed: %s", exc)
        return None


def compute_text_generation(
    file_path: str | Path,
    bert_model: str = DEFAULT_BERT_MODEL,
) -> dict[str, Any] | None:
    """Evaluate a JSON file containing ``human_answer`` and ``assistant_answer`` pairs.

    Args:
        file_path: Path to JSON file (list of objects).
        bert_model: BERT model for BERTScore.

    Returns:
        Dictionary of metric scores, or ``None`` on error.
    """
    file_path = Path(file_path)
    if not file_path.is_file():
        logger.error("File not found: %s", file_path)
        return None

    references: list[str] = []
    candidates: list[str] = []

    try:
        data = json.loads(file_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        logger.error("Invalid JSON in %s: %s", file_path, exc)
        return None

    for idx, item in enumerate(data):
        human = item.get("human_answer", "") or "None"
        assistant = item.get("assistant_answer", "") or "None"
        if not item.get("human_answer"):
            logger.warning("Item %d has empty human_answer", idx)
        if not item.get("assistant_answer"):
            logger.warning("Item %d has empty assistant_answer", idx)
        references.append(tokenize_chinese(human))
        candidates.append(tokenize_chinese(assistant))

    if not references:
        logger.error("No valid data found in %s", file_path)
        return None

    logger.info("Evaluating %d samples", len(references))

    rouge_1, rouge_2, rouge_l = calculate_rouge_score(references, candidates)
    bleu_1, bleu_4, bleu_l = calculate_bleu_scores(references, candidates)
    meteor = calculate_meteor_score(references, candidates)
    cider = calculate_cider_score(references, candidates)
    bert_f1 = calculate_bert_score(references, candidates, model_type=bert_model)

    return {
        "rouge-1": rouge_1,
        "rouge-2": rouge_2,
        "rouge-l": rouge_l,
        "bleu-1": bleu_1,
        "bleu-4": bleu_4,
        "bleu-l": bleu_l,
        "meteor": meteor,
        "cider": cider,
        "bert_f1": bert_f1,
    }


def _demo_evaluate() -> None:
    """Run evaluation on built-in demo data."""
    sample_data = [
        {
            "human_answer": "6月27日20时至28日20时，河北南部、山东西北部...",
            "assistant_answer": "6月27日20时至28日20时，华北中东部、黄淮中东部...",
        },
        {
            "human_answer": "7月8日14时至9日14时，浙江中南部和东部...",
            "assistant_answer": "7月8日14时至9日14时，华北中南部、黄淮西部...",
        },
    ]

    references = [tokenize_chinese(item["human_answer"]) for item in sample_data]
    candidates = [tokenize_chinese(item["assistant_answer"]) for item in sample_data]

    calculate_bleu_scores(references, candidates)
    calculate_rouge_score(references, candidates)
    calculate_meteor_score(references, candidates)
    calculate_cider_score(references, candidates)
    calculate_bert_score(references, candidates)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate text generation with traditional NLG metrics."
    )
    parser.add_argument("--file", type=str, required=True, help="JSON evaluation data file")
    parser.add_argument(
        "--output_path",
        type=str,
        default=".",
        help="Directory to save evaluation_summary.json",
    )
    parser.add_argument(
        "--bert_model",
        type=str,
        default=DEFAULT_BERT_MODEL,
        help="BERT model path or name",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run on demo data instead of file",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if args.demo:
        _demo_evaluate()
        return

    result = compute_text_generation(args.file, bert_model=args.bert_model)
    if result is None:
        raise SystemExit(1)

    out_dir = Path(args.output_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "evaluation_summary.json"
    out_file.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info("Results saved to %s", out_file)


if __name__ == "__main__":
    main()

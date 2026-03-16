"""
Prepare GoogleNews word2vec embedding.

What this script does:
1) Loads a word2vec binary model (GoogleNews-vectors-negative300.bin.gz)
2) Standardizes token lookup with fallback rules
3) Returns L2-normalized vectors
4) Computes coverage stats for definitional pairs and occupations lists
5) Saves a JSON coverage report

Usage example:
python src/prepare_embedding.py \
  --embedding_path /path/to/GoogleNews-vectors-negative300.bin.gz \
  --def_pairs data/definitional_pairs.txt \
  --occupations data/occupations.txt \
  --output_dir outputs/prep
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    from gensim.models import KeyedVectors
except ImportError as e:
    raise ImportError(
        "gensim is required. Install with: pip install gensim"
    ) from e

# Data structures
@dataclass
class TokenLookupResult:
    original: str
    found_token: Optional[str]
    strategy: Optional[str]
    in_vocab: bool

# Embedding wrapper
class EmbeddingHelper:
    """
    Wraps gensim KeyedVectors with:
      - token canonicalization / fallback lookup
      - L2-normalized vector access
      - coverage stats helpers
    """

    def __init__(self, kv: KeyedVectors):
        self.kv = kv
        self.vector_size = kv.vector_size

    @classmethod
    def load_word2vec(cls, path: str, binary: bool = True) -> "EmbeddingHelper":
        if not os.path.exists(path):
            raise FileNotFoundError(f"Embedding file not found: {path}")
        print(f"[INFO] Loading embedding from: {path}")
        kv = KeyedVectors.load_word2vec_format(path, binary=binary)
        print(f"[INFO] Loaded embedding. Vocab size: {len(kv.key_to_index):,}")
        print(f"[INFO] Vector dim: {kv.vector_size}")
        return cls(kv)

    def _candidates(self, word: str) -> List[Tuple[str, str]]:
        """
        Generate fallback candidates in order.
        Returns list of (candidate_token, strategy_name).
        """
        w = word.strip()
        if not w:
            return []

        candidates: List[Tuple[str, str]] = []
        seen = set()

        def add(token: str, strategy: str) -> None:
            if token and token not in seen:
                seen.add(token)
                candidates.append((token, strategy))

        # Original
        add(w, "exact")

        # Multi-word phrase variants
        if " " in w:
            add(w.replace(" ", "_"), "spaces_to_underscores")

        # Lowercase variants
        add(w.lower(), "lower")
        if " " in w:
            add(w.lower().replace(" ", "_"), "lower_spaces_to_underscores")

        # Title-case variants (sometimes useful for names / GoogleNews casing)
        add(w.title(), "title")
        if " " in w:
            add(w.title().replace(" ", "_"), "title_spaces_to_underscores")

        # Upper-case variant (rarely useful but cheap)
        add(w.upper(), "upper")
        if " " in w:
            add(w.upper().replace(" ", "_"), "upper_spaces_to_underscores")

        return candidates

    def lookup_token(self, word: str) -> TokenLookupResult:
        for token, strategy in self._candidates(word):
            if token in self.kv.key_to_index:
                return TokenLookupResult(
                    original=word,
                    found_token=token,
                    strategy=strategy,
                    in_vocab=True,
                )
        return TokenLookupResult(
            original=word,
            found_token=None,
            strategy=None,
            in_vocab=False,
        )

    def get_vector(self, word: str, normalize: bool = True) -> Optional[np.ndarray]:
        result = self.lookup_token(word)
        if not result.in_vocab or result.found_token is None:
            return None

        v = np.array(self.kv[result.found_token], dtype=np.float64)
        if normalize:
            norm = np.linalg.norm(v)
            if norm == 0:
                return None
            v = v / norm
        return v

# File readers
def read_nonempty_lines(path: str) -> List[str]:
    lines: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if line.startswith("#"):
                continue
            lines.append(line)
    return lines

def read_definitional_pairs(path: str) -> List[Tuple[str, str]]:
    """
    Supported formats per line:
      he,she
      he\tshe
      he she
    """
    pairs: List[Tuple[str, str]] = []
    for line in read_nonempty_lines(path):
        if "," in line:
            parts = [p.strip() for p in line.split(",")]
        elif "\t" in line:
            parts = [p.strip() for p in line.split("\t")]
        else:
            parts = line.split()

        if len(parts) != 2:
            raise ValueError(
                f"Expected exactly 2 tokens in definitional pair line, got: {line}"
            )
        pairs.append((parts[0], parts[1]))
    return pairs


def read_word_list(path: str) -> List[str]:
    return read_nonempty_lines(path)

# Coverage checks
def coverage_for_words(
    helper: EmbeddingHelper,
    words: Sequence[str],
) -> Dict:
    rows = []
    found = 0
    strategy_counts: Dict[str, int] = {}

    for w in words:
        r = helper.lookup_token(w)
        if r.in_vocab:
            found += 1
            strategy_counts[r.strategy or "unknown"] = (
                strategy_counts.get(r.strategy or "unknown", 0) + 1
            )
        rows.append(
            {
                "original": r.original,
                "in_vocab": r.in_vocab,
                "found_token": r.found_token,
                "strategy": r.strategy,
            }
        )

    total = len(words)
    coverage = found / total if total > 0 else math.nan

    return {
        "total": total,
        "found": found,
        "missing": total - found,
        "coverage": coverage,
        "strategy_counts": strategy_counts,
        "rows": rows,
    }

def coverage_for_pairs(
    helper: EmbeddingHelper,
    pairs: Sequence[Tuple[str, str]],
) -> Dict:
    rows = []
    both_found = 0
    left_found = 0
    right_found = 0

    for a, b in pairs:
        ra = helper.lookup_token(a)
        rb = helper.lookup_token(b)

        if ra.in_vocab:
            left_found += 1
        if rb.in_vocab:
            right_found += 1
        if ra.in_vocab and rb.in_vocab:
            both_found += 1

        rows.append(
            {
                "left_original": a,
                "left_in_vocab": ra.in_vocab,
                "left_found_token": ra.found_token,
                "left_strategy": ra.strategy,
                "right_original": b,
                "right_in_vocab": rb.in_vocab,
                "right_found_token": rb.found_token,
                "right_strategy": rb.strategy,
                "pair_usable": ra.in_vocab and rb.in_vocab,
            }
        )

    total = len(pairs)
    return {
        "total_pairs": total,
        "left_found": left_found,
        "right_found": right_found,
        "both_found": both_found,
        "pair_coverage": (both_found / total) if total > 0 else math.nan,
        "rows": rows,
    }

def run_sanity_checks(helper: EmbeddingHelper, words: Sequence[str]) -> Dict:
    out = {}
    for w in words:
        vec = helper.get_vector(w, normalize=True)
        r = helper.lookup_token(w)
        if vec is None:
            out[w] = {
                "in_vocab": False,
                "found_token": r.found_token,
                "strategy": r.strategy,
                "norm_after_normalization": None,
                "vector_dim": None,
            }
        else:
            out[w] = {
                "in_vocab": True,
                "found_token": r.found_token,
                "strategy": r.strategy,
                "norm_after_normalization": float(np.linalg.norm(vec)),
                "vector_dim": int(vec.shape[0]),
            }
    return out

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding_path", type=str, required=True)
    parser.add_argument(
        "--binary",
        action="store_true",
        help="Set this flag if loading word2vec binary format (.bin/.bin.gz).",
    )
    parser.add_argument("--def_pairs", type=str, required=True)
    parser.add_argument("--occupations", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Most GoogleNews files are binary. If user forgot --binary, infer from suffix.
    binary = args.binary or args.embedding_path.endswith(".bin") or args.embedding_path.endswith(".bin.gz")

    helper = EmbeddingHelper.load_word2vec(args.embedding_path, binary=binary)

    definitional_pairs = read_definitional_pairs(args.def_pairs)
    occupations = read_word_list(args.occupations)

    print(f"[INFO] Definitional pairs loaded: {len(definitional_pairs)}")
    print(f"[INFO] Occupation words loaded: {len(occupations)}")

    pair_cov = coverage_for_pairs(helper, definitional_pairs)
    occ_cov = coverage_for_words(helper, occupations)

    sanity_words = ["he", "she", "man", "woman", "doctor", "nurse", "engineer", "teacher"]
    sanity = run_sanity_checks(helper, sanity_words)

    report = {
        "embedding": {
            "path": args.embedding_path,
            "binary": binary,
            "vocab_size": len(helper.kv.key_to_index),
            "vector_dim": helper.vector_size,
        },
        "preprocessing_policy": {
            "token_lookup_fallback_order": [
                "exact",
                "spaces_to_underscores",
                "lower",
                "lower_spaces_to_underscores",
                "title",
                "title_spaces_to_underscores",
                "upper",
                "upper_spaces_to_underscores",
            ],
            "oov_policy": "exclude OOV words from downstream lists; report coverage",
            "vector_normalization": "L2 normalize every vector before subspace/bias computations",
        },
        "coverage": {
            "definitional_pairs": pair_cov,
            "occupations": occ_cov,
        },
        "sanity_checks": sanity,
    }

    report_path = os.path.join(args.output_dir, "coverage_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"[INFO] Saved coverage report to: {report_path}")

    # Also save compact text summary for quick viewing
    summary_path = os.path.join(args.output_dir, "coverage_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("=== Embedding ===\n")
        f.write(f"Path: {args.embedding_path}\n")
        f.write(f"Vocab size: {len(helper.kv.key_to_index):,}\n")
        f.write(f"Vector dim: {helper.vector_size}\n\n")

        f.write("=== Definitional Pairs Coverage ===\n")
        f.write(f"Total pairs: {pair_cov['total_pairs']}\n")
        f.write(f"Both found: {pair_cov['both_found']}\n")
        f.write(f"Pair coverage: {pair_cov['pair_coverage']:.4f}\n\n")

        f.write("=== Occupations Coverage ===\n")
        f.write(f"Total words: {occ_cov['total']}\n")
        f.write(f"Found: {occ_cov['found']}\n")
        f.write(f"Missing: {occ_cov['missing']}\n")
        f.write(f"Coverage: {occ_cov['coverage']:.4f}\n")
        f.write(f"Strategy counts: {occ_cov['strategy_counts']}\n\n")

        f.write("=== Sanity Checks ===\n")
        for w, info in sanity.items():
            f.write(f"{w}: {info}\n")

    print(f"[INFO] Saved text summary to: {summary_path}")
    print("[DONE] Preparation step completed successfully.")

if __name__ == "__main__":
    main()
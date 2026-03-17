"""
Compute gender subspace

Method:
1) Load embedding and definitional pairs.
2) Keep only pairs with both words in-vocabulary.
3) For each pair (w1, w2), center around pair mean:
     mu = (v1 + v2) / 2
     c1 = v1 - mu
     c2 = v2 - mu
4) Run PCA on stacked centered vectors.
5) Use PC1 as gender direction; top-k PCs as gender subspace.

Outputs:
- gender_subspace_report.json
- gender_direction.npy
- gender_subspace_components.npy
"""

from __future__ import annotations
import argparse
import json
import os
from typing import Dict, List, Tuple
import numpy as np
from prepare_embedding import EmbeddingHelper, read_definitional_pairs

def usable_pairs(
    helper: EmbeddingHelper, pairs: List[Tuple[str, str]]
    ) -> List[Tuple[str, str, str, str]]:
    rows: List[Tuple[str, str, str, str]] = []
    for left, right in pairs:
        l = helper.lookup_token(left)
        r = helper.lookup_token(right)
        if l.in_vocab and r.in_vocab and l.found_token and r.found_token:
            rows.append((left, right, l.found_token, r.found_token))
    return rows

def build_centered_matrix(
    helper: EmbeddingHelper, pairs: List[Tuple[str, str, str, str]]
    ) -> np.ndarray:
    centered: List[np.ndarray] = []
    for _, _, tok_left, tok_right in pairs:
        v_left = helper.get_vector(tok_left, normalize=True)
        v_right = helper.get_vector(tok_right, normalize=True)
        if v_left is None or v_right is None:
            continue
        mu = (v_left + v_right) / 2.0
        centered.append(v_left - mu)
        centered.append(v_right - mu)
    if not centered:
        raise ValueError("No valid centered vectors produced from definitional pairs.")
    return np.stack(centered, axis=0)

def pca_subspace(matrix: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    # X shape: [n_samples, dim]
    # SVD: X = U S Vt, rows of Vt are principal directions.
    _, singular_values, vt = np.linalg.svd(matrix, full_matrices=False)
    components = vt[:k]
    # Explained variance ratio from singular values.
    eigenvalues = (singular_values ** 2) / max(matrix.shape[0] - 1, 1)
    total = float(np.sum(eigenvalues))
    if total == 0.0:
        explained_ratio = np.zeros_like(eigenvalues)
    else:
        explained_ratio = eigenvalues / total
    return components, explained_ratio

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding_path", type=str, required=True)
    parser.add_argument(
        "--binary",
        action="store_true",
        help="Set for word2vec binary files (.bin/.bin.gz).",
    )
    parser.add_argument("--def_pairs", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument(
        "--k",
        type=int,
        default=1,
        help="Number of principal components for gender subspace.",
    )
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    if args.k <= 0:
        raise ValueError("--k must be >= 1")

    os.makedirs(args.output_dir, exist_ok=True)
    binary = (
        args.binary
        or args.embedding_path.endswith(".bin")
        or args.embedding_path.endswith(".bin.gz")
    )

    helper = EmbeddingHelper.load_word2vec(args.embedding_path, binary=binary)
    pairs = read_definitional_pairs(args.def_pairs)
    usable = usable_pairs(helper, pairs)
    if not usable:
        raise ValueError("No usable definitional pairs (both tokens in vocab).")

    x = build_centered_matrix(helper, usable)
    k = min(args.k, x.shape[1], x.shape[0])
    components, explained_ratio = pca_subspace(x, k=k)
    direction = components[0]

    direction_path = os.path.join(args.output_dir, "gender_direction.npy")
    subspace_path = os.path.join(args.output_dir, "gender_subspace_components.npy")
    report_path = os.path.join(args.output_dir, "gender_subspace_report.json")

    np.save(direction_path, direction)
    np.save(subspace_path, components)

    report: Dict = {
        "embedding_path": args.embedding_path,
        "binary": binary,
        "vector_dim": int(helper.vector_size),
        "total_definitional_pairs": len(pairs),
        "usable_definitional_pairs": len(usable),
        "usable_pairs": [
            {
                "left_original": lo,
                "right_original": ro,
                "left_token": lt,
                "right_token": rt,
            }
            for lo, ro, lt, rt in usable
        ],
        "centered_matrix_shape": [int(x.shape[0]), int(x.shape[1])],
        "k_used": int(k),
        "explained_variance_ratio_top_k": [float(v) for v in explained_ratio[:k]],
        "gender_direction_norm": float(np.linalg.norm(direction)),
        "gender_direction_first_10_dims": [float(v) for v in direction[:10]],
        "outputs": {
            "gender_direction_npy": direction_path,
            "gender_subspace_components_npy": subspace_path,
        },
        "notes": {
            "method": "PCA on centered definitional pair vectors (Bolukbasi et al., 2016).",
            "direction": "First principal component (PC1).",
            "subspace": "Span of top-k principal components.",
        },
    }

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"[INFO] Usable definitional pairs: {len(usable)}/{len(pairs)}")
    print(f"[INFO] Centered matrix shape: {x.shape}")
    print(f"[INFO] k used: {k}")
    print(f"[INFO] Saved direction to: {direction_path}")
    print(f"[INFO] Saved subspace to: {subspace_path}")
    print(f"[INFO] Saved report to: {report_path}")

if __name__ == "__main__":
    main()

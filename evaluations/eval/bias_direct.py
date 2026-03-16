import numpy as np
from typing import List
from eval.embedding import DebiasedEmbedding

def direct_bias(emb: DebiasedEmbedding, neutral_words: List[str], gender_dir: np.ndarray, c: float = 1.0) -> float:
    g = gender_dir / (np.linalg.norm(gender_dir) + 1e-12)
    vals = []
    for w in neutral_words:
        v = emb.vec(w)
        if v is None:
            continue
        vals.append(abs(float(np.dot(v, g))) ** c)
    if not vals:
        raise ValueError("No valid neutral words for DirectBias.")
    return float(np.mean(vals))

def load_word_list(path: str) -> List[str]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            w = line.strip()
            if w:
                out.append(w.lower())
    return out
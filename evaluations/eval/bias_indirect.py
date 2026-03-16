from typing import List, Tuple

import numpy as np
from eval.embedding import DebiasedEmbedding

def _unit(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x) + 1e-12)

def _cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

def beta_gender_portion(emb: DebiasedEmbedding, w: str, v: str, gender_dir: np.ndarray) -> float:
    """
    β(w,v): fraction of similarity between w and v that is due to gender direction.
    Compute using cosine similarity:
        beta = (cos(w,v) - cos(w_perp, v_perp)) / (cos(w,v) + eps)
    where w_perp = w - proj_g(w), v_perp = v - proj_g(v).
    """
    vw, vv = emb.vec(w), emb.vec(v)
    if vw is None or vv is None:
        return float("nan")

    g = _unit(gender_dir)

    # project onto gender direction
    w_g = float(np.dot(vw, g)) * g
    v_g = float(np.dot(vv, g)) * g
    w_perp = vw - w_g
    v_perp = vv - v_g

    c_full = _cos(vw, vv)
    c_perp = _cos(w_perp, v_perp)

    return float((c_full - c_perp) / (c_full + 1e-12))

def extremes_on_axis(emb: DebiasedEmbedding, words: List[str], pos: str, neg: str, topk: int = 20) -> Tuple[List[Tuple[str,float]], List[Tuple[str,float]]]:
    vpos, vneg = emb.vec(pos), emb.vec(neg)
    if vpos is None or vneg is None:
        raise ValueError("pos/neg word missing in embedding.")
    axis = _unit(vpos - vneg)
    scored = []
    for w in words:
        vw = emb.vec(w)
        if vw is None:
            continue
        scored.append((w, float(np.dot(vw, axis))))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:topk], scored[-topk:]
import numpy as np
from typing import List, Tuple
from eval.embedding import DebiasedEmbedding

def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

def solve_analogy_topk(emb: DebiasedEmbedding, a: str, b: str, c: str, candidates: List[str], k: int = 5):
    va, vb, vc = emb.vec(a), emb.vec(b), emb.vec(c)
    if va is None or vb is None or vc is None:
        return []
    target = vb - va + vc
    scored = []
    for w in candidates:
        if w in {a,b,c}: 
            continue
        vw = emb.vec(w)
        if vw is None:
            continue
        scored.append((w, cosine(vw, target)))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:k]

import os
import json

def load_stereotype_lexicon(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(obj, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

def save_csv(rows, header, path: str):
    import csv
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)

def norm(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v) + 1e-12)

def label_pair_with_lexicon(x: str, y: str, lex: dict):
    """
    用确定性词表/启发式替代 MTurk。
    返回: (stereotype_label, appropriate_label) in {"yes","no","unknown"}
    """
    x = x.lower(); y = y.lower()

    stereo_pairs = set(tuple(map(str.lower, p)) for p in lex.get("stereotype_pairs", []))
    appr_pairs   = set(tuple(map(str.lower, p)) for p in lex.get("appropriate_pairs", []))

    if (x, y) in stereo_pairs or (y, x) in stereo_pairs:
        return "yes", "no"
    if (x, y) in appr_pairs or (y, x) in appr_pairs:
        return "no", "yes"

    fem  = set(w.lower() for w in lex.get("female_coded", []))
    male = set(w.lower() for w in lex.get("male_coded", []))
    if (x in fem and y in male) or (x in male and y in fem):
        return "yes", "unknown"

    return "unknown", "unknown"

def generate_gender_analogy_pairs(
    emb: DebiasedEmbedding,
    a: str = "she",
    b: str = "he",
    topn: int = 150,
    delta: float = 1.0,
    x_pool: int = 5000,
):
    """
    近似复现论文 Section 4: 给定 seed(a,b)，生成 top-N 对 (x,y) 使得 a:x :: b:y。
    用 cosAdd 提议 y≈x+b-a，并用 cos(seed_dir, x-y) 打分，同时约束 ||x-y||<=delta。
    返回: List[(x,y,score,dist)]
    """
    va = emb.vec(a); vb = emb.vec(b)
    if va is None or vb is None:
        raise ValueError(f"Seed words not found: {a}, {b}")

    seed_dir = norm(va - vb)

    # 选一批候选 x，避免 runtime 爆炸
    vocab = emb.vocab()
    xs = []
    for w in vocab:
        vw = emb.vec(w)
        if vw is None:
            continue
        xs.append((w, float(np.dot(norm(vw), seed_dir))))
    xs.sort(key=lambda t: t[1], reverse=True)
    pool = [w for w, _ in xs[:x_pool]]

    pairs = []
    used_x = set()

    for x in pool:
        if x in used_x:
            continue
        vx = emb.vec(x)
        if vx is None:
            continue

        target = vx + vb - va
        y = emb.most_similar_to_vec(target, exclude={a, b, x})
        if y is None:
            continue

        vy = emb.vec(y)
        if vy is None:
            continue

        dist = float(np.linalg.norm(vx - vy))
        if dist > delta:
            continue

        score = cosine(seed_dir, norm(vx - vy))
        if score <= 0:
            continue

        pairs.append((x, y, float(score), dist))
        used_x.add(x)

        if len(pairs) >= topn:
            break

    pairs.sort(key=lambda t: t[2], reverse=True)
    return pairs[:topn]

def run_gender_analogy_probe(
    emb_before: DebiasedEmbedding,
    emb_after: DebiasedEmbedding,
    out_dir: str,
    lexicon_path: str = "data/stereotype_lexicon.json",
    topn: int = 150,
    delta: float = 1.0,
):
    """
    输出文件：
      - gender_analogies_before.csv
      - gender_analogies_after.csv
      - gender_analogy_summary.json
    """
    os.makedirs(out_dir, exist_ok=True)
    lex = load_stereotype_lexicon(lexicon_path)

    pairs_b = generate_gender_analogy_pairs(emb_before, "she", "he", topn=topn, delta=delta)
    pairs_a = generate_gender_analogy_pairs(emb_after,  "she", "he", topn=topn, delta=delta)

    def summarize(pairs):
        stereo = appr = unk = 0
        rows = []
        for x, y, s, d in pairs:
            ls, la = label_pair_with_lexicon(x, y, lex)
            if ls == "yes":
                stereo += 1
            elif la == "yes":
                appr += 1
            else:
                unk += 1
            rows.append([x, y, s, d, ls, la])
        n = len(pairs)
        return {
            "n": n,
            "stereotype_yes": stereo,
            "appropriate_yes": appr,
            "unknown": unk,
            "stereotype_rate": stereo / n if n else 0.0,
            "appropriate_rate": appr / n if n else 0.0,
            "rows": rows,
        }

    sb = summarize(pairs_b)
    sa = summarize(pairs_a)

    save_csv(
        sb["rows"],
        ["x", "y", "score", "dist", "stereotype", "appropriate"],
        os.path.join(out_dir, "gender_analogies_before.csv"),
    )
    save_csv(
        sa["rows"],
        ["x", "y", "score", "dist", "stereotype", "appropriate"],
        os.path.join(out_dir, "gender_analogies_after.csv"),
    )

    summary = {
        "seed": ["she", "he"],
        "topn": topn,
        "delta": delta,
        "lexicon_path": lexicon_path,
        "before": {k: sb[k] for k in ["n","stereotype_yes","appropriate_yes","unknown","stereotype_rate","appropriate_rate"]},
        "after":  {k: sa[k] for k in ["n","stereotype_yes","appropriate_yes","unknown","stereotype_rate","appropriate_rate"]},
        "notes": "Paper uses MTurk for stereotype labels; here replaced with deterministic lexicon/heuristic for reproducibility."
    }
    save_json(summary, os.path.join(out_dir, "gender_analogy_summary.json"))
    return summary
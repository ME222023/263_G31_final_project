import numpy as np
from scipy.stats import spearmanr
from typing import List, Tuple
from eval.embedding import DebiasedEmbedding

def cosine(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:
    return float(np.dot(a, b) / (max(np.linalg.norm(a), eps) * max(np.linalg.norm(b), eps)))

def eval_word_similarity(
    emb: DebiasedEmbedding,
    pairs: List[Tuple[str, str, float]],
) -> float:
    gold, pred = [], []
    for w1, w2, score in pairs:
        v1, v2 = emb.vec(w1), emb.vec(w2)
        if v1 is None or v2 is None:
            continue
        gold.append(score)
        pred.append(cosine(v1, v2))
    if len(gold) < 10:
        raise ValueError("Too few valid pairs (OOV too many).")
    return float(spearmanr(gold, pred).correlation)

def load_rg65(path: str) -> List[Tuple[str, str, float]]:
    # 常见格式：word1 word2 score（空格或tab）
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: 
                continue
            w1, w2, s = line.split()
            out.append((w1.lower(), w2.lower(), float(s)))
    return out

def load_ws353(path: str) -> List[Tuple[str, str, float]]:
    # 常见格式：word1<TAB>word2<TAB>score
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.lower().startswith("word1"):  # header
                continue
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            w1, w2, s = parts[0], parts[1], parts[2]
            out.append((w1.lower(), w2.lower(), float(s)))
    return out

def analogy_accuracy_msr(emb: DebiasedEmbedding, questions: List[Tuple[str,str,str,str]]) -> float:
    """
    3CosAdd / cosAdd:
      find d* = argmax_{w!=a,b,c} cos(w, vec(b)-vec(a)+vec(c))
    """
    correct, total = 0, 0
    # 注意：全词表暴力argmax很慢；这里用 gensim 的 most_similar_by_vector 更快。
    # 我们需要一个 KeyedVectors 来做近邻搜索；若你的 modified 很少，可以直接用原模型近邻，再用 emb.vec() 重算分数做 rerank。
    kv = emb.orig  # fallback：用原始索引近邻候选
    for a,b,c,d in questions:
        va, vb, vc = emb.vec(a), emb.vec(b), emb.vec(c)
        if va is None or vb is None or vc is None:
            continue
        target = (vb - va + vc)
        # 取 topN 候选，再用“混合向量接口”重算 cosine 选最大
        candidates = [w for (w,_) in kv.most_similar(positive=[b,c], negative=[a], topn=50)]
        best_w, best_s = None, -1e9
        for w in candidates:
            if w in {a,b,c}:
                continue
            vw = emb.vec(w)
            if vw is None:
                continue
            s = float(np.dot(vw, target) / (np.linalg.norm(target) + 1e-12))
            if s > best_s:
                best_s, best_w = s, w
        if best_w is None:
            continue
        total += 1
        if best_w == d:
            correct += 1
    return correct / max(total, 1)

def load_msr_analogy(path: str) -> List[Tuple[str,str,str,str]]:
    # 常见格式：a b c d（每行四词），可能有 ": section" 行
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith(":"):
                continue
            a,b,c,d = line.split()
            out.append((a.lower(), b.lower(), c.lower(), d.lower()))
    return out
import numpy as np
from typing import Dict, Optional
from gensim.models import KeyedVectors


class DebiasedEmbedding:
    """
    Query vectors from:
      - original GoogleNews word2vec (gensim KeyedVectors)
      - overridden by modified vectors from .npz (hard-debias output)
    """
    def __init__(self, original: KeyedVectors, modified: Dict[str, np.ndarray]):
        self.orig = original
        # store modified vectors normalized (same as orig.vec() returns)
        self.modified = {k: self._unit(v) for k, v in modified.items()}

    @staticmethod
    def _unit(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
        n = np.linalg.norm(v)
        return v / max(n, eps)

    def has(self, w: str) -> bool:
        return (w in self.modified) or (w in self.orig.key_to_index)

    def vec(self, w: str) -> Optional[np.ndarray]:
        """
        Return a unit-normalized vector for word w, or None if OOV.
        Note: keep the original token casing behavior as-is.
        """
        if w in self.modified:
            return self.modified[w]
        if w in self.orig.key_to_index:
            return self._unit(self.orig.get_vector(w))
        return None

    def vocab(self):
        """
        Return vocabulary list from the original KeyedVectors.
        """
        return list(self.orig.key_to_index.keys())

    def most_similar_to_vec(self, vec, exclude=None, topn=50):
        """
        Return the most similar word to the given vector, excluding some tokens.
        Uses gensim KeyedVectors most_similar with positive=[vec].
        """
        exclude = exclude or set()
        # normalize exclude tokens for robust filtering
        exclude_l = set(w.lower() for w in exclude)

        # gensim most_similar may return original-cased tokens
        sims = self.orig.most_similar(positive=[vec], topn=topn)
        for w, _ in sims:
            wl = w.lower()
            if wl in exclude_l:
                continue
            return wl
        return None


def load_modified_npz(npz_path: str) -> Dict[str, np.ndarray]:
    data = np.load(npz_path, allow_pickle=True)
    # 兼容不同保存格式：tokens+vectors 或 dict
    if "tokens" in data and "vectors" in data:
        tokens = data["tokens"]
        vecs = data["vectors"]
        return {str(t): vecs[i] for i, t in enumerate(tokens)}
    if "modified" in data:
        d = data["modified"].item()
        return {str(k): np.array(v) for k, v in d.items()}
    # fallback: 假设每个 key 是 token
    out = {}
    for k in data.files:
        out[str(k)] = np.array(data[k])
    return out
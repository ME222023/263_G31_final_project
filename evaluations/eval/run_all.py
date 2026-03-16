import os
import sys
import json
import time
from typing import List, Tuple
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import matplotlib.pyplot as plt
from gensim.models import KeyedVectors

from eval.embedding import DebiasedEmbedding, load_modified_npz
from eval.utility import load_msr_analogy, eval_word_similarity, analogy_accuracy_msr
from eval.bias_direct import load_word_list, direct_bias
from eval.bias_indirect import extremes_on_axis, beta_gender_portion
from eval.bias_analogy_probe import run_gender_analogy_probe


# -----------------------------
# Logging helpers
# -----------------------------
T0 = time.time()

def log(msg: str):
    """Print log with elapsed time; flush to show immediately."""
    dt = time.time() - T0
    print(f"[{dt:8.1f}s] {msg}", flush=True)

class timed:
    """Context manager to time a step."""
    def __init__(self, name: str):
        self.name = name
        self.t_start = None

    def __enter__(self):
        self.t_start = time.time()
        log(f"START: {self.name}")
        return self

    def __exit__(self, exc_type, exc, tb):
        t = time.time() - self.t_start
        if exc_type is None:
            log(f"END  : {self.name} (took {t:.1f}s)")
        else:
            log(f"FAIL : {self.name} (after {t:.1f}s) -> {exc_type.__name__}: {exc}")
        return False  # do not swallow exceptions


# -----------------------------
# Helpers: IO + data loading
# -----------------------------
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def save_json(obj, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

def save_csv(rows, header, path: str):
    import csv
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)

def load_similarity_any(path: str):
    """
    Load RG65 / WordSim353 from csv/tsv/txt into List[(w1, w2, score)].
    Supports CSV with header (common variants) or whitespace-separated 3 cols.
    """
    import csv

    def norm(h: str) -> str:
        return h.strip().lower().replace(" ", "").replace("_", "")

    with open(path, "r", encoding="utf-8", newline="") as f:
        sample = f.read(4096)
        f.seek(0)

        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",\t;")
            has_header = csv.Sniffer().has_header(sample)
        except Exception:
            dialect = None
            has_header = False

        rows = []

        if dialect is not None:
            reader = csv.reader(f, dialect)
            data = list(reader)
            if not data:
                return rows

            if has_header:
                header = data[0]
                hmap = {norm(h): i for i, h in enumerate(header)}

                def pick(keys):
                    for k in keys:
                        kk = norm(k)
                        if kk in hmap:
                            return hmap[kk]
                    return None

                i1 = pick(["word1", "word 1", "w1", "term1"])
                i2 = pick(["word2", "word 2", "w2", "term2"])
                iscore = pick(["score", "similarity", "relatedness", "human(mean)", "humanmean", "mean", "gold", "rating"])

                if i1 is None or i2 is None or iscore is None:
                    raise ValueError(f"Cannot find word1/word2/score columns in {path}. header={header}")

                for r in data[1:]:
                    if len(r) <= max(i1, i2, iscore):
                        continue
                    w1 = r[i1].strip().lower()
                    w2 = r[i2].strip().lower()
                    try:
                        s = float(r[iscore])
                    except Exception:
                        continue
                    if w1 and w2:
                        rows.append((w1, w2, s))
                return rows

            # no header: assume 3 cols
            for r in data:
                if len(r) < 3:
                    continue
                w1, w2 = r[0].strip().lower(), r[1].strip().lower()
                try:
                    s = float(r[2])
                except Exception:
                    continue
                if w1 and w2:
                    rows.append((w1, w2, s))
            return rows

        # fallback whitespace
        f.seek(0)
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            w1, w2 = parts[0].lower(), parts[1].lower()
            try:
                s = float(parts[2])
            except Exception:
                continue
            rows.append((w1, w2, s))
        return rows


# -----------------------------
# Plot helpers
# -----------------------------
def plot_beta_bars(words, beta_before, beta_after, title, out_path: str):
    x = np.arange(len(words))
    width = 0.38
    plt.figure(figsize=(12, 4.5))
    plt.bar(x - width / 2, beta_before, width, label="before")
    plt.bar(x + width / 2, beta_after,  width, label="hard-debiased")
    plt.xticks(x, words, rotation=60, ha="right")
    plt.ylabel("β (gender portion of similarity)")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

def plot_hist_before_after(values_before, values_after, title, xlabel, out_path: str, bins=30):
    plt.figure(figsize=(8, 4.5))
    plt.hist(values_before, bins=bins, alpha=0.6, label="before")
    plt.hist(values_after,  bins=bins, alpha=0.6, label="hard-debiased")
    plt.xlabel(xlabel)
    plt.ylabel("count")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

def plot_box_before_after(values_before, values_after, title, ylabel, out_path: str):
    plt.figure(figsize=(6, 4.5))
    plt.boxplot([values_before, values_after], labels=["before", "hard-debiased"], showfliers=True)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


# -----------------------------
# Utility helpers for v2 outputs
# -----------------------------
def safe_float(x):
    try:
        if x is None:
            return float("nan")
        return float(x)
    except Exception:
        return float("nan")

def finite_list(xs):
    xs = [safe_float(x) for x in xs]
    return [x for x in xs if np.isfinite(x)]

def summarize_distribution(xs):
    xs = finite_list(xs)
    if not xs:
        return {"n": 0}
    arr = np.array(xs, dtype=float)
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "std": float(arr.std()),
        "p05": float(np.quantile(arr, 0.05)),
        "p25": float(np.quantile(arr, 0.25)),
        "p75": float(np.quantile(arr, 0.75)),
        "p95": float(np.quantile(arr, 0.95)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


# -----------------------------
# Main
# -----------------------------
def main():
    log("Program started.")

    # ===== paths =====
    google_bin = "GoogleNews-vectors-negative300.bin.gz"
    modified_npz = "outputs/debias/debiased_modified_vectors.npz"
    gender_dir_path = "outputs/prep/gender_direction.npy"
    occupations_path = "data/occupations.txt"

    rg_path = "data/rg65.csv"
    ws_path = "data/wordsim353.csv"
    msr_path = "data/questions-words.txt"

    out_dir = "outputs/results"
    ensure_dir(out_dir)
    log(f"Output dir ready: {out_dir}")

    # ===== load embeddings =====
    with timed("Load GoogleNews KeyedVectors (this can take minutes on first run)"):
        log(f"Loading embedding from: {google_bin}")
        orig = KeyedVectors.load_word2vec_format(google_bin, binary=True)
        log(f"Loaded embedding: vocab={len(orig.key_to_index):,}, dim={orig.vector_size}")

    with timed("Load modified vectors (.npz)"):
        log(f"Loading modified npz from: {modified_npz}")
        modified = load_modified_npz(modified_npz)
        log(f"Loaded modified tokens: {len(modified):,}")

    with timed("Wrap embeddings (before/after)"):
        emb_before = DebiasedEmbedding(orig, {})
        emb_after  = DebiasedEmbedding(orig, modified)

    with timed("Load gender direction + occupations"):
        gender_dir = np.load(gender_dir_path)
        occ = load_word_list(occupations_path)
        log(f"Loaded gender_dir shape={gender_dir.shape}, occupations={len(occ)}")

    # ===== Utility =====
    with timed("Load similarity benchmarks (RG65/WS353)"):
        rg = load_similarity_any(rg_path)
        ws = load_similarity_any(ws_path)
        log(f"Loaded RG65 pairs={len(rg)}, WS353 pairs={len(ws)}")

    with timed("Load MSR analogy benchmark"):
        msr = load_msr_analogy(msr_path)
        log(f"Loaded MSR analogy questions={len(msr)}")

    with timed("Compute Utility scores (RG65/WS353/MSR)"):
        rg_b = eval_word_similarity(emb_before, rg)
        rg_a = eval_word_similarity(emb_after, rg)
        ws_b = eval_word_similarity(emb_before, ws)
        ws_a = eval_word_similarity(emb_after, ws)

        # MSR analogy can be slower
        log("Computing MSR analogy BEFORE ...")
        msr_b = analogy_accuracy_msr(emb_before, msr)
        log("Computing MSR analogy AFTER ...")
        msr_a = analogy_accuracy_msr(emb_after, msr)

        log(f"Utility done. RG65 {rg_b:.4f}->{rg_a:.4f}, WS353 {ws_b:.4f}->{ws_a:.4f}, MSR {msr_b:.4f}->{msr_a:.4f}")

    with timed("Save Utility table (original outputs)"):
        utility_rows = [
            ["GoogleNews",           rg_b, ws_b, msr_b],
            ["Hard-debiased",        rg_a, ws_a, msr_a],
        ]
        save_csv(
            rows=utility_rows,
            header=["Setting", "RG65_spearman", "WS353_spearman", "MSR_analogy_acc"],
            path=os.path.join(out_dir, "utility_table.csv")
        )
        save_json(
            {
                "setting": {
                    "before": {"RG65_spearman": rg_b, "WS353_spearman": ws_b, "MSR_analogy_acc": msr_b},
                    "hard_debiased": {"RG65_spearman": rg_a, "WS353_spearman": ws_a, "MSR_analogy_acc": msr_a},
                }
            },
            os.path.join(out_dir, "utility_table.json")
        )

    # ===== Direct Bias (original outputs) =====
    with timed("Compute DirectBias c=1 (original)"):
        db_b = direct_bias(emb_before, occ, gender_dir, c=1.0)
        db_a = direct_bias(emb_after,  occ, gender_dir, c=1.0)
        log(f"DirectBias_c1 {db_b:.6g}->{db_a:.6g}")

    with timed("Save bias_metrics.json (original output)"):
        bias_obj = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "DirectBias_c1": {"before": db_b, "hard_debiased": db_a},
            "notes": {"neutral_words_source": occupations_path, "gender_direction": gender_dir_path}
        }
        save_json(bias_obj, os.path.join(out_dir, "bias_metrics.json"))

    # ===== Indirect Bias (original outputs) =====
    topk = 20
    with timed(f"Compute extremes_on_axis (topk={topk}, original)"):
        soft_extremes, foot_extremes = extremes_on_axis(
            emb_before, occ, "softball", "football", topk=topk
        )

    with timed("Compute beta for extremes + save indirect CSVs (original)"):
        softball_rows, softball_words, beta_soft_before, beta_soft_after = [], [], [], []
        for w, proj in soft_extremes:
            b0 = beta_gender_portion(emb_before, w, "softball", gender_dir)
            b1 = beta_gender_portion(emb_after,  w, "softball", gender_dir)
            softball_rows.append([w, proj, b0, b1])
            softball_words.append(w)
            beta_soft_before.append(b0)
            beta_soft_after.append(b1)

        football_rows, football_words, beta_foot_before, beta_foot_after = [], [], [], []
        for w, proj in foot_extremes:
            b0 = beta_gender_portion(emb_before, w, "football", gender_dir)
            b1 = beta_gender_portion(emb_after,  w, "football", gender_dir)
            football_rows.append([w, proj, b0, b1])
            football_words.append(w)
            beta_foot_before.append(b0)
            beta_foot_after.append(b1)

        save_csv(
            rows=softball_rows,
            header=["word", "axis_projection_softball_minus_football", "beta_before", "beta_after"],
            path=os.path.join(out_dir, "indirect_softball.csv")
        )
        save_csv(
            rows=football_rows,
            header=["word", "axis_projection_softball_minus_football", "beta_before", "beta_after"],
            path=os.path.join(out_dir, "indirect_football.csv")
        )

    with timed("Plot beta_topk bars (original)"):
        plot_beta_bars(
            words=softball_words,
            beta_before=beta_soft_before,
            beta_after=beta_soft_after,
            title=f"Indirect bias toward softball (Top-{topk} extremes defined by BEFORE)",
            out_path=os.path.join(out_dir, "beta_softball_topk.png")
        )
        plot_beta_bars(
            words=football_words,
            beta_before=beta_foot_before,
            beta_after=beta_foot_after,
            title=f"Indirect bias toward football (Top-{topk} extremes defined by BEFORE)",
            out_path=os.path.join(out_dir, "beta_football_topk.png")
        )

    # ============================================================
    # >>> NEW: Indirect bias extremes defined by AFTER (add-only) <<<
    # ============================================================
    with timed(f"[ADD] Compute extremes_on_axis on AFTER (topk={topk})"):
        soft_extremes_after, foot_extremes_after = extremes_on_axis(
            emb_after, occ, "softball", "football", topk=topk
        )

    with timed("[ADD] Compute beta for AFTER-extremes + save CSVs"):
        softball_rows_after, softball_words_after = [], []
        beta_soft_before_after, beta_soft_after_after = [], []

        for w, proj in soft_extremes_after:
            b0 = beta_gender_portion(emb_before, w, "softball", gender_dir)
            b1 = beta_gender_portion(emb_after,  w, "softball", gender_dir)
            softball_rows_after.append([w, proj, b0, b1])
            softball_words_after.append(w)
            beta_soft_before_after.append(b0)
            beta_soft_after_after.append(b1)

        football_rows_after, football_words_after = [], []
        beta_foot_before_after, beta_foot_after_after = [], []

        for w, proj in foot_extremes_after:
            b0 = beta_gender_portion(emb_before, w, "football", gender_dir)
            b1 = beta_gender_portion(emb_after,  w, "football", gender_dir)
            football_rows_after.append([w, proj, b0, b1])
            football_words_after.append(w)
            beta_foot_before_after.append(b0)
            beta_foot_after_after.append(b1)

        save_csv(
            rows=softball_rows_after,
            header=["word", "axis_projection_softball_minus_football", "beta_before", "beta_after"],
            path=os.path.join(out_dir, "indirect_softball_after_extremes.csv")
        )
        save_csv(
            rows=football_rows_after,
            header=["word", "axis_projection_softball_minus_football", "beta_before", "beta_after"],
            path=os.path.join(out_dir, "indirect_football_after_extremes.csv")
        )

    with timed("[ADD] Plot beta_topk bars (extremes defined by AFTER)"):
        plot_beta_bars(
            words=softball_words_after,
            beta_before=beta_soft_before_after,
            beta_after=beta_soft_after_after,
            title=f"Indirect bias toward softball (Top-{topk} extremes defined by AFTER)",
            out_path=os.path.join(out_dir, "beta_softball_topk_after_extremes.png")
        )
        plot_beta_bars(
            words=football_words_after,
            beta_before=beta_foot_before_after,
            beta_after=beta_foot_after_after,
            title=f"Indirect bias toward football (Top-{topk} extremes defined by AFTER)",
            out_path=os.path.join(out_dir, "beta_football_topk_after_extremes.png")
        )
    # ============================================================

    # ===== Gender analogy stereotype probe (Fig2/Fig8 style) =====
    summary = run_gender_analogy_probe(
        emb_before=emb_before,
        emb_after=emb_after,
        out_dir=out_dir,  # 你已有的 outputs/results
        lexicon_path="data/stereotype_lexicon.json",
        topn=150,
        delta=1.0
    )

    print(
        f"[gender-analogy] stereotype_rate {summary['before']['stereotype_rate']:.3f}"
        f" -> {summary['after']['stereotype_rate']:.3f}, "
        f"appropriate_rate {summary['before']['appropriate_rate']:.3f}"
        f" -> {summary['after']['appropriate_rate']:.3f}"
    )

    # ============================================================
    # V2 NEW OUTPUTS (do not overwrite previous ones)
    # ============================================================

    with timed("V2-1 DirectBias for c in {1,2,3}"):
        direct_by_c = []
        direct_by_c_json = {}
        for c in [1.0, 2.0, 3.0]:
            v0 = direct_bias(emb_before, occ, gender_dir, c=c)
            v1 = direct_bias(emb_after,  occ, gender_dir, c=c)
            direct_by_c.append([c, v0, v1])
            direct_by_c_json[str(c)] = {"before": v0, "hard_debiased": v1}

        save_csv(
            rows=direct_by_c,
            header=["c", "DirectBias_before", "DirectBias_hard_debiased"],
            path=os.path.join(out_dir, "v2_direct_bias_by_c.csv")
        )
        save_json(
            {"DirectBias_by_c": direct_by_c_json},
            os.path.join(out_dir, "v2_direct_bias_by_c.json")
        )

    with timed("V2-2 Rank most gendered occupations + plots"):
        def cos(a, b):
            return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

        g = gender_dir / (np.linalg.norm(gender_dir) + 1e-12)

        rank_rows_before, rank_rows_after = [], []
        proj_abs_before, proj_abs_after = [], []

        for w in occ:
            vb = emb_before.vec(w)
            va = emb_after.vec(w)
            if vb is None or va is None:
                continue
            p0 = abs(cos(vb, g))
            p1 = abs(cos(va, g))
            proj_abs_before.append(p0)
            proj_abs_after.append(p1)
            rank_rows_before.append([w, p0])
            rank_rows_after.append([w, p1])

        rank_rows_before.sort(key=lambda x: x[1], reverse=True)
        rank_rows_after.sort(key=lambda x: x[1], reverse=True)

        topn = 50
        save_csv(
            rows=rank_rows_before[:topn],
            header=["word", "abs_cos_to_gender_dir"],
            path=os.path.join(out_dir, f"v2_most_gendered_before_top{topn}.csv")
        )
        save_csv(
            rows=rank_rows_after[:topn],
            header=["word", "abs_cos_to_gender_dir"],
            path=os.path.join(out_dir, f"v2_most_gendered_after_top{topn}.csv")
        )

        plot_hist_before_after(
            values_before=proj_abs_before,
            values_after=proj_abs_after,
            title="Distribution of |cos(w, gender_dir)| over occupations",
            xlabel="|cos(w, gender_dir)|",
            out_path=os.path.join(out_dir, "v2_gender_projection_hist.png"),
            bins=30
        )
        plot_box_before_after(
            values_before=proj_abs_before,
            values_after=proj_abs_after,
            title="Boxplot of |cos(w, gender_dir)| over occupations",
            ylabel="|cos(w, gender_dir)|",
            out_path=os.path.join(out_dir, "v2_gender_projection_box.png")
        )

    with timed("V2-3 Indirect beta distribution over ALL occupations + stats/plots"):
        beta_soft_all_before, beta_soft_all_after = [], []
        beta_foot_all_before, beta_foot_all_after = [], []

        for w in occ:
            beta_soft_all_before.append(beta_gender_portion(emb_before, w, "softball", gender_dir))
            beta_soft_all_after.append(beta_gender_portion(emb_after,  w, "softball", gender_dir))
            beta_foot_all_before.append(beta_gender_portion(emb_before, w, "football", gender_dir))
            beta_foot_all_after.append(beta_gender_portion(emb_after,  w, "football", gender_dir))

        beta_stats = {
            "beta_softball": {
                "before": summarize_distribution(beta_soft_all_before),
                "hard_debiased": summarize_distribution(beta_soft_all_after),
            },
            "beta_football": {
                "before": summarize_distribution(beta_foot_all_before),
                "hard_debiased": summarize_distribution(beta_foot_all_after),
            }
        }
        save_json(beta_stats, os.path.join(out_dir, "v2_beta_distribution_stats.json"))

        plot_hist_before_after(
            finite_list(beta_soft_all_before),
            finite_list(beta_soft_all_after),
            title="β distribution over occupations (anchor=softball)",
            xlabel="β(w, softball)",
            out_path=os.path.join(out_dir, "v2_beta_softball_hist.png"),
            bins=30
        )
        plot_hist_before_after(
            finite_list(beta_foot_all_before),
            finite_list(beta_foot_all_after),
            title="β distribution over occupations (anchor=football)",
            xlabel="β(w, football)",
            out_path=os.path.join(out_dir, "v2_beta_football_hist.png"),
            bins=30
        )
        plot_box_before_after(
            finite_list(beta_soft_all_before),
            finite_list(beta_soft_all_after),
            title="β boxplot over occupations (anchor=softball)",
            ylabel="β(w, softball)",
            out_path=os.path.join(out_dir, "v2_beta_softball_box.png")
        )
        plot_box_before_after(
            finite_list(beta_foot_all_before),
            finite_list(beta_foot_all_after),
            title="β boxplot over occupations (anchor=football)",
            ylabel="β(w, football)",
            out_path=os.path.join(out_dir, "v2_beta_football_box.png")
        )

    with timed("V2-4 Save manifest"):
        manifest = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "paths": {
                "google_bin": google_bin,
                "modified_npz": modified_npz,
                "gender_dir_path": gender_dir_path,
                "occupations_path": occupations_path,
                "rg_path": rg_path,
                "ws_path": ws_path,
                "msr_path": msr_path,
            },
            "settings": {
                "topk_extremes": topk,
                "topn_most_gendered": 50,
                "direct_bias_c_list": [1.0, 2.0, 3.0],
            }
        }
        save_json(manifest, os.path.join(out_dir, "v2_manifest.json"))

    # ===== Summary =====
    log("All done. Key outputs:")
    log(f"  utility_table.csv/json, bias_metrics.json, indirect_*.csv, beta_*_topk.png")
    log(f"  plus v2_* extras in: {out_dir}")


if __name__ == "__main__":
    main()
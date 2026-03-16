import re
import random
from collections import defaultdict

RAW_PATH = "data/occupations.txt"
OUT_PATH = "data/occupations_350.txt"
TARGET_N = 350
SEED = 263

# 这些是常见“标题/类别/垃圾词”的模式，可按需要增减
BLOCK_PATTERNS = [
    r"\ball occupation\b",
    r"\boccupation\b$",
    r"\bworker\b$",
    r"\bspecialist\b$",
    r"\btechnician\b$",      # 可选：如果你想保留 technician，就删掉这一条
    r"\bteacher\b$",         # 可选：如果你想保留 teacher，就删掉这一条
    r"\bmanager\b$",         # 可选
]

def clean_line(s: str) -> str:
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    s = s.strip("•-–—,;:")
    if not s:
        return ""

    # 去掉非常短的碎片（如 Bu / Coache）
    if len(s) <= 2:
        return ""

    # 去掉含明显“分类”字样的行（你也可以更严格）
    low = s.lower()
    for pat in BLOCK_PATTERNS:
        if re.search(pat, low):
            return ""

    # 丢掉只有一个非常泛的词（这些在你的 raw 里经常是碎片）
    if low in {"tender", "assistant", "operator", "repairer", "installer", "clerk", "technician"}:
        return ""

    # 统一成 googlenews 常见形式：小写 + 空格变下划线
    s = low.replace("/", "_")
    s = s.replace(" ", "_")

    # 去掉连续下划线
    s = re.sub(r"_+", "_", s)

    # 只保留字母/下划线（去掉括号等）
    s = re.sub(r"[^a-z_]", "", s).strip("_")

    # 清洗后仍过短就丢
    if len(s) < 3:
        return ""

    return s

def main():
    random.seed(SEED)

    # 1) 读取 + 清洗
    items = []
    with open(RAW_PATH, "r", encoding="utf-8") as f:
        for line in f:
            x = clean_line(line)
            if x:
                items.append(x)

    # 2) 去重（保持顺序）
    seen = set()
    uniq = []
    for x in items:
        if x not in seen:
            seen.add(x)
            uniq.append(x)

    if len(uniq) < TARGET_N:
        raise ValueError(f"Not enough cleaned items: {len(uniq)} < {TARGET_N}. "
                         f"Consider relaxing BLOCK_PATTERNS.")

    # 3) 为了不偏某一字母：按首字母分桶，做一个“近似均匀”的抽样
    buckets = defaultdict(list)
    for x in uniq:
        buckets[x[0]].append(x)

    letters = sorted(buckets.keys())
    per_bucket = TARGET_N // len(letters)
    chosen = []

    # 先每桶拿 per_bucket 个
    for ch in letters:
        pool = buckets[ch]
        random.shuffle(pool)
        chosen.extend(pool[:per_bucket])

    # 不够就从剩余里补齐
    remaining = []
    chosen_set = set(chosen)
    for ch in letters:
        for x in buckets[ch]:
            if x not in chosen_set:
                remaining.append(x)
    random.shuffle(remaining)
    while len(chosen) < TARGET_N:
        chosen.append(remaining.pop())

    chosen = chosen[:TARGET_N]
    chosen.sort()  # 输出再按字母排序，方便看

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for x in chosen:
            f.write(x + "\n")

    print(f"[OK] cleaned unique: {len(uniq)}")
    print(f"[OK] wrote {len(chosen)} occupations to {OUT_PATH}")

if __name__ == "__main__":
    main()
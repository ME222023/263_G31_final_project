import os
import pandas as pd
import requests

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def save_rg65_ws353(out_dir="data"):
    """
    Export:
      - data/RG65.txt  (word1 word2 score)
      - data/WS353.txt (word1 word2 score)
    """
    ensure_dir(out_dir)

    # mangoes provides RG65 & WS353 datasets
    from mangoes.dataset import Dataset  # type: ignore

    rg = Dataset("RG65").data  # pandas-like table
    ws = Dataset("WS353").data

    # Normalize columns to: word1 word2 score
    def normalize(df: pd.DataFrame):
        cols = [c.lower() for c in df.columns]
        df = df.copy()
        df.columns = cols
        # common variants: word1/word2/score OR "word 1"/"word 2"/"human (mean)"
        if "word 1" in cols: df = df.rename(columns={"word 1": "word1"})
        if "word 2" in cols: df = df.rename(columns={"word 2": "word2"})
        if "human (mean)" in cols: df = df.rename(columns={"human (mean)": "score"})
        return df[["word1", "word2", "score"]]

    rg = normalize(pd.DataFrame(rg))
    ws = normalize(pd.DataFrame(ws))

    rg_path = os.path.join(out_dir, "RG65.txt")
    ws_path = os.path.join(out_dir, "WS353.txt")

    rg.to_csv(rg_path, sep=" ", index=False, header=False)
    ws.to_csv(ws_path, sep=" ", index=False, header=False)

    print("[OK] wrote:", rg_path)
    print("[OK] wrote:", ws_path)

def download_questions_words(out_dir="data"):
    """
    Download analogy questions file used by Mikolov word2vec release:
      data/questions-words.txt
    """
    ensure_dir(out_dir)
    url = "https://raw.githubusercontent.com/tmikolov/word2vec/master/questions-words.txt"
    out_path = os.path.join(out_dir, "questions-words.txt")

    r = requests.get(url, timeout=30)
    r.raise_for_status()
    with open(out_path, "wb") as f:
        f.write(r.content)

    print("[OK] wrote:", out_path)

if __name__ == "__main__":
    save_rg65_ws353("data")
    download_questions_words("data")
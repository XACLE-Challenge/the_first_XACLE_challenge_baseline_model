import pandas as pd
from pathlib import Path

ROOT = Path("data_list_for_benchmark_model")
filelist = ["train.txt", "val.txt", "ev.txt"]

# 入力ファイルの列名
COLS = ["listener", "wav", "score", "model", "caption", "_"]

def load_table(path: Path) -> pd.DataFrame:
    # ヘッダーがないので names=COLS を指定
    return pd.read_csv(path, sep=None, engine="python", names=COLS)

def pick_caption(series: pd.Series) -> str:
    """wav ごとに caption を1つ決める（最頻値、なければ最初）"""
    s = series.dropna()
    if s.empty:
        return ""
    m = s.mode()
    return (m.iloc[0] if not m.empty else s.iloc[0])

def process_file(in_path: Path, round_digits: int = 3):
    df = load_table(in_path)
    # score を数値化
    df["score"] = pd.to_numeric(df["score"], errors="coerce")
    
    agg = (
        df.groupby("wav", as_index=False)
          .agg(
              MOS=("score", "mean"),
              caption=("caption", pick_caption),
          )
    )
    agg = agg[["wav", "caption", "MOS"]]
    agg["MOS"] = agg["MOS"].round(round_digits)
    
    # 出力ファイル名：元ファイル名の末尾に _mos.txt を付与
    out_path = in_path.with_name(in_path.stem + "_mos.txt")
    agg.to_csv(out_path, sep=",", index=False)  # ← カンマ区切り
    print(f"Saved: {out_path}")

if __name__ == "__main__":
    for fname in filelist:
        process_file(ROOT / fname)

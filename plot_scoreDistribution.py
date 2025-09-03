import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_score_stats(txt_file_path: str, title: str, save_dir: str = "score_plots"):
    # Create directory if not exists
    os.makedirs(save_dir, exist_ok=True)

    # Load CSV
    df = pd.read_csv(txt_file_path, header=None, names=["listener", "wav", "score", "model", "caption", "_"])
    df["score"] = pd.to_numeric(df["score"], errors="coerce")
    df = df.dropna(subset=["score"])

    # --- Overall Score Histogram ---
    plt.figure(figsize=(8, 4))
    sns.histplot(df["score"], bins=11, binrange=(0, 10), kde=False)
    plt.title(f"{title} - Overall Score Distribution")
    plt.xlabel("Score")
    plt.ylabel("Count")
    plt.xticks(range(0, 11))
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{title}_overall.png"))
    plt.close()

    # --- Score Histogram by Model ---
    plt.figure(figsize=(10, 5))
    sns.histplot(data=df, x="score", hue="model", bins=11, binrange=(0, 10), multiple="dodge", shrink=0.8)
    plt.title(f"{title} - Score Distribution by Model")
    plt.xlabel("Score")
    plt.ylabel("Count")
    plt.xticks(range(0, 11))
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{title}_by_model.png"))
    plt.close()

    # --- Statistics Output ---
    print(f"\n📊 Score Statistics: {title}")
    print("■ Overall:")
    print(df["score"].describe())

    print("\n■ By Model:")
    print(df.groupby("model")["score"].describe())

# === Example Usage ===
train_txt = "./datasets/data/data_list_for_benchmark_model/train.txt"
val_txt   = "./datasets/data/data_list_for_benchmark_model/val.txt"
plot_score_stats(train_txt, "Train")
plot_score_stats(val_txt, "Validation")


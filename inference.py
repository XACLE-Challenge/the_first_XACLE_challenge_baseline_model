import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from collections.abc import Mapping, Sequence
from transformers.tokenization_utils_base import BatchEncoding
from models.Byola import get_normalizer
from models.xacle_benchmark_model import XACLEBenchmarkModel
from datasets.xacle_benchmark_dataset import get_dataset
import utils.utils as utils
from tqdm import tqdm
import csv
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import spearmanr, pearsonr

RESULT_CSV = "./result/inference_results.csv"
RESULT_PLT = "./result/scatterplot.png"

def scatter_plot():
    df = pd.read_csv(RESULT_CSV)
    df["model"] = df["wav_path"].apply(lambda x: x.split("/")[4])
    gt = df["gt_mos"].values
    pred = df["pred_mos"].values
    models = df["model"].values

    plt.figure(figsize=(6,6))
    for model_name in sorted(set(models)):
        mask = models == model_name
        plt.scatter(
            gt[mask],
            pred[mask],
            label=model_name,
            alpha=0.7
        )

    min_val = min(gt.min(), pred.min())
    max_val = max(gt.max(), pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], "k--", lw=1)
    plt.xlabel("Ground Truth MOS")
    plt.ylabel("Predicted MOS")
    plt.title("Predicted vs. Ground Truth MOS")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(RESULT_PLT, dpi=200)

    srcc, _ = spearmanr(gt, pred)
    lcc, _ = pearsonr(gt, pred)
    print(f"SRCC: {srcc:.4f}")
    print(f"LCC : {lcc:.4f}")


def move_to_device(obj, device):
    """
    Recursively move tensors (and common container objects that hold tensors)
    to the target device.

    Parameters
    ----------
    obj : Any
        Arbitrary object that may contain tensors (Tensor, dict-like, list-like,
        or transformers.BatchEncoding). Non-tensor scalars/strings are left unchanged.
    device : torch.device or str
        Target device (e.g., "cuda:0", "cpu").

    Returns
    -------
    Any
        Same structure as `obj`, but with all tensors placed on `device`.
    """
    # 1) Single tensor → move directly
    if isinstance(obj, torch.Tensor):
        return obj.to(device)

    # 2) HuggingFace BatchEncoding supports .to(), handle it explicitly
    if isinstance(obj, BatchEncoding):
        return obj.to(device)

    # 3) Mapping types (dict, UserDict, etc.) → recurse on each value
    if isinstance(obj, Mapping):
        return {k: move_to_device(v, device) for k, v in obj.items()}

    # 4) Sequence types (list, tuple, etc.) → preserve container type
    if isinstance(obj, Sequence) and not isinstance(obj, (str, bytes)):
        return type(obj)(move_to_device(v, device) for v in obj)

    # 5) Anything else (int, float, str, Path, ...) → leave untouched
    return obj

def inference(cfg):
    device = torch.device(cfg["device"])

    # -------- tokenizer / dataset / dataloader --------
    tokenizer = AutoTokenizer.from_pretrained(cfg["roberta"]["pretrained_model"], cache_dir="./hf_cache")
    test_ds   = get_dataset(
        cfg["test_list"],
        cfg["wav_dir"],
        tokenizer=tokenizer,
        max_sec=cfg["max_len"],
        sr=cfg["byola"]["sample_rate"],
        org_max=10.0,
        org_min=0.0
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=1,
        shuffle=False,
        num_workers=cfg["num_workers"],
        collate_fn=test_ds.collate_fn
    )
    # -------------------------------------------------

    # -------- model / normalizer --------
    model = XACLEBenchmarkModel(cfg, device).to(device)
    chkpt = torch.load(os.path.join(cfg["output_dir"], "best_model.pt"), map_location=device)
    model.load_state_dict(chkpt, strict=True)
    model.eval()
    normalizer = get_normalizer(cfg["byola"], "test_list")
    # ------------------------------------

    # -------- run inference --------
    rows = []
    with torch.no_grad():
        for batch in tqdm(test_loader):
            batch = move_to_device(batch, device)
            pred  = model.forward(batch, normalizer)
            pred  = pred.detach().cpu().item()
            pred_mos = pred * 5.0 + 5.0
            gt = batch["scores"].cpu().item()
            gt_mos = gt * 5.0 + 5.0
            rows.append({
                "wav_path" : batch["wav_paths"][0],
                "pred_mos": round(pred_mos,2),
                "gt_mos": round(gt_mos, 2)
            })

    with open(RESULT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["wav_path", "pred_mos", "gt_mos"])
        writer.writeheader()
        writer.writerows(rows)
    # -------------------------------

    # ------- plot a scatter diagram -------
    scatter_plot()
    # --------------------------------------

    return

if __name__ == "__main__":
    config = utils.load_config("config.json")
    inference(config)
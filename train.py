import os
import torch
import numpy as np
import time
from torch.utils.data import DataLoader
from scipy.stats import spearmanr
from transformers import AutoTokenizer
from collections.abc import Mapping, Sequence
from transformers.tokenization_utils_base import BatchEncoding

from models.Byola import get_normalizer
from models.xacle_benchmark_model import XACLEBenchmarkModel
from datasets.xacle_benchmark_dataset import get_dataset
from losses.loss_function import get_loss_function

import utils.utils as utils

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

def train(cfg):
    os.makedirs(cfg["output_dir"], exist_ok=True)
    device = torch.device(cfg["device"])

    # -------- tokenizer / dataset / dataloader --------
    tokenizer = AutoTokenizer.from_pretrained(cfg["roberta"]["pretrained_model"], cache_dir="./hf_cache")

    train_ds = get_dataset(
        cfg["train_list"],
        cfg["wav_dir"],
        tokenizer=tokenizer,
        max_sec=cfg["max_len"],
        sr=cfg["byola"]["sample_rate"],
        org_max=10.0,
        org_min=0.0
    )
    val_ds   = get_dataset(
        cfg["val_list"],
        cfg["wav_dir"],
        tokenizer=tokenizer,
        max_sec=cfg["max_len"],
        sr=cfg["byola"]["sample_rate"],
        org_max=10.0,
        org_min=0.0
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=cfg["num_workers"],
        collate_fn=train_ds.collate_fn,
        drop_last=True
    )
    val_loader = DataLoader(
        val_ds, 
        batch_size=cfg["val_batch_size"], 
        shuffle=False,
        num_workers=cfg["num_workers"], 
        collate_fn=val_ds.collate_fn
    )
    # -------------------------------------------------

    # -------- model / loss / opt --------
    model = XACLEBenchmarkModel(cfg, device).to(device)
    loss_fn  = get_loss_function(cfg["loss"])
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt,
        mode="min",
        factor=0.5,
        patience=5
    )
    print("Prepare normalizer")
    normalizer_train = get_normalizer(cfg["byola"], "train_list")
    normalizer_eval  = get_normalizer(cfg["byola"], "val_list")
    best_srcc, patience = -np.inf, 0
    # ------------------------------------
    print("train loader:",len(train_loader))
    for epoch in range(cfg["epochs"]):
        model.train()
        start_time = time.time()
        epoch_loss = 0.0
        for batch in train_loader:
            # ---------- setup ----------
            batch = move_to_device(batch, device)
            opt.zero_grad()
            # ---------------------------

            # ---------- forward ----------
            pred = model.forward(batch, normalizer_train)   # (B, )
            # -----------------------------

            # ---------- loss ----------
            loss = loss_fn(pred, batch["scores"], batch["num_class"])
            loss.backward()
            opt.step()
            epoch_loss += loss.item()
            # --------------------------

        # ---------- Train Epoch loss ----------  
        avg_train_loss = epoch_loss / len(train_loader)
        # --------------------------------------

        ################################################################
        train_pred = (pred.detach().cpu().numpy() * 5 + 5).tolist()
        train_gt   = (batch["scores"].detach().cpu().numpy() * 5 + 5).tolist()
        ################################################################
        
        # ---------- Validation ----------
        model.eval()
        val_loss = 0.0
        preds, gts = [], []
        with torch.no_grad():
            for batch in val_loader:
                batch = move_to_device(batch, device)
                pred  = model.forward(batch, normalizer_eval)
                loss = loss_fn(pred, batch["scores"], batch["num_class"])
                val_loss += loss.item()
                preds.extend((pred.cpu().numpy() * 5 + 5).tolist())
                gts.extend((batch["scores"].cpu().numpy() * 5 + 5).tolist())
        avg_val_loss = val_loss / len(val_loader)
        srcc = spearmanr(gts, preds).correlation
        mse  = np.mean((np.array(gts) - np.array(preds)) ** 2)
        # --------------------------------

        ################################################################
        val_pred = (pred.detach().cpu().numpy() * 5 + 5).tolist()
        val_gt   = (batch["scores"].detach().cpu().numpy() * 5 + 5).tolist()
        ################################################################

        # ---------- timer ----------
        end_time = time.time()
        elapsed_time = end_time - start_time
        # ---------------------------
    
        # ---------- epoch summary ----------
        print(f"Epoch {epoch:04d} completed in {elapsed_time:.2f} seconds | Train Loss : {avg_train_loss:.4f}\tVal Loss : {avg_val_loss:.4f}\tVal SRCC / MSE: {srcc:.4f} , {mse:.4f}")
        # -----------------------------------

        ########################################################################
        train_pred = [f"{v: 05.2f}" for v in train_pred]
        train_gt   = [f"{v: 05.2f}" for v in train_gt]
        val_pred   = [f"{v: 05.2f}" for v in val_pred]
        val_gt     = [f"{v: 05.2f}" for v in val_gt]
        print(f"\ttrain pred   : {train_pred}")
        print(f"\ttrain_gt     : {train_gt}")
        print(f"\tval pred     : {val_pred}")
        print(f"\tval_gt       : {val_gt}")
        ########################################################################

        # ---------- Scheduler ----------
        scheduler.step(avg_val_loss)
        # -------------------------------

        # ---------- early-stop ----------
        if srcc > best_srcc:
            best_srcc, patience = srcc, 0
            torch.save(model.state_dict(), os.path.join(cfg["output_dir"], "best_model.pt"))
            print("✅  best model updated")
        else:
            patience += 1
            if patience >= cfg["early_stop_patience"]:
                print("⛔️ Early stopping (patience exhausted)")
                break
        # --------------------------------

    return


if __name__ == "__main__":
    config = utils.load_config("config.json")
    train(config)



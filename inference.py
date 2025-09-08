import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from models.Byola import get_normalizer
from models.xacle_benchmark_model import XACLEBenchmarkModel
from datasets.xacle_benchmark_dataset import get_infdataset
import utils.utils as utils
from tqdm import tqdm
import csv
import sys
import os

def inference():
    # -------- initial setup --------
    if len(sys.argv) < 2:
        print("Usage: ptyhon inference.py chkpt_dir_name")
        sys.exit(1)
    chkpt_dir = os.path.join("./chkpt", sys.argv[1])
    if not os.path.isdir(chkpt_dir):
        print(f"Error: CheckPoint Directory {chkpt_dir} does not exist.")
        sys.exit(1)
    chkpt_path = os.path.join(chkpt_dir, "best_model.pt")
    cfg_path   = os.path.join(chkpt_dir, "config.json")
    if not os.path.isfile(chkpt_path):
        print(f"Error: Expected CheckPoint does not exist.")
        sys.exit(1)
    if not os.path.isfile(cfg_path):
        print(f"Error: Expected Config file does not exist")
    if len(sys.argv) == 2:
        dataset_key = "validation"
    elif sys.argv[2] == "test":
        dataset_key = "test"
    else:
        print("Error: Specify the evaluation dataset using the third command-line arguments.")
    cfg = utils.load_config(cfg_path)
    dataset_label   = f"{dataset_key}_list"
    dataset_list    = cfg[dataset_label]
    dataset_wav_dir = os.path.join(cfg["wav_dir"], dataset_key)
    print("Perform inference on the following dataset with following checkpoint.")
    print(f"\tchkpt:        {chkpt_path}")
    print(f"\tdata list:    {dataset_list}")
    print(f"\twav dir:      {dataset_wav_dir}") 
    device = torch.device(cfg["device"])
    # -------------------------------

    # -------- tokenizer / dataset / dataloader --------
    tokenizer = AutoTokenizer.from_pretrained(cfg["text_encoder"]["pretrained_model"], cache_dir="./hf_cache")
    test_ds   = get_infdataset(
        txt_file_path=dataset_list,
        wav_dir=dataset_wav_dir,
        tokenizer=tokenizer,
        max_sec=cfg["max_len"],
        sr=cfg["audio_encoder"]["sample_rate"]
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
    chkpt = torch.load(chkpt_path, map_location=device)
    model.load_state_dict(chkpt, strict=True)
    model.eval()
    normalizer = get_normalizer(cfg, dataset_label)
    # ------------------------------------

    # -------- run inference --------
    rows = []
    with torch.no_grad():
        for batch in tqdm(test_loader):
            batch = utils.move_to_device(batch, device)
            pred  = model.forward(batch, normalizer)
            pred  = pred.detach().cpu().item()
            pred_mos = pred * 5.0 + 5.0
            rows.append({
                "wav_file_name" : os.path.basename(batch["wav_paths"][0]),
                "pred_score": round(pred_mos,2)
            })
    # -------------------------------

    # -------- write results --------
    result_path = os.path.join(chkpt_dir, f"inference_result_for_{dataset_key}.csv")
    print(f"Inference has completed. Results will be written to the following file: \n\t{result_path}")
    with open(result_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["wav_file_name", "pred_score"])
        writer.writeheader()
        writer.writerows(rows)
    # -------------------------------

    return

if __name__ == "__main__":
    inference()
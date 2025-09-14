## 📌 Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Evaluation Code](#evaluation-code)
- [Result](#result)
- [License](#license)
- [Citation](#citation)
- 

<h2 id="overview">📖 Overview</h2>

This repository contains the baseline model for automatic evaluation of text–audio alignment in [the first XACLE challenge](https://xacle.org/index.html). It provides a model trained to estimate subjective evaluation scores from text–audio pairs. In the baseline model, BYOL-A is used as the audio encoder and RoBERTa as the text encoder, and score prediction is performed using the features extracted from these encoders. <br>We sincerely thank the authors for sharing the official code and facilitating the advancement of academia. <br><br>
<p align="center">
<img src="pics/task_overview.png" alt="Overview of score prediction of audio–text alignment" width="400">
</p>
<h2 id="features">✨ Features</h2>

- Automatically evaluates text–audio alignment scores.
- Supports BYOL-A (Audio Encoder) and RoBERTa (Text Encoder)
- Provides ready-to-use trained baseline model.

<h2 id="requirements">💻 Requirements</h2>

- Python : Tested on 3.9.21
- CUDA   : Tested on 11.8 (for GPU acceleration)
- Python Packages (the core dependencies for this project are listed in requirements.txt)
    - easydict==1.13
    - librosa==0.9.2
    - matplotlib==3.5.1
    - nnAudio==0.3.3
    - numpy==1.24.1
    - pandas==1.4.1
    - pytorch-lightning==1.6.0
    - scikit-learn==1.0.2
    - SoundFile==0.10.3.post1
    - tqdm==4.66.5
    - transformers==4.15.0

<h2 id="installation">⚙️ Installation</h2>

### 1. Clone the repository
```bash
git clone https://github.com/XACLE-Challenge/XACLE2026_baseline_model.git
```
```bash
cd cloned-repository
```
### 2. Create and activate a virtual environment
```bash
python -m venv xacle_env && source xacle_env/bin/activate
```

### 3. Upgrade pip
```bash
python -m pip install --upgrade "pip<24.1"
```

### 4. Install required packages
```bash
pip install -r requirements.txt
```

### 5. Install Torch
Please install Torch and Torchaudio according to your environment. Below is an example of the version that has been confirmed to work in a CUDA 11.8 environment.
```bash
pip install torch==2.2.0+cu118 torchaudio==2.2.0+cu118 \
  --index-url https://download.pytorch.org/whl/cu118
```

### 6. Download datasets and a pretrained baseline model
- **Datasets**
  - To download the dataset, please send an email to [dataset@xacle.org](mailto:dataset@xacle.org) with “**Download dataset**” in the subject line. You will receive an automated reply with the download link.


  - After downloading the dataset, please place it in the **datasets** directory.
- **A prerained baseline model**
  - A pretrained baseline model can be downloaded from [here](https://y-okamoto.sakura.ne.jp/XACLE_Challenge/2025/baseline_model/trained_baseline_model.zip)
  - For details about the files contained within
    - *best_model.pt* : The saved model
    - *config.json* : The configuration file used for training
    - *inference_result_for_validation.csv* : Inference results on the validation data using *best_model.pt*
    - *log.txt* : Standart output results during training
    - *metricts_result_for_validation.csv* : Evaluation results of the inference outputs in *inference_result_for_validation.csv*
  - After downloading a pretrained model, please place it in the **chkpt** directory.

- **Regarding the placement of these directories, please refer to the [Project Structure](#project-structure).**

<h2 id="project-structure">📂 Project Structure</h2>

```bash
xacle2026_baseline_model/
├── README.md
├── LICENSE
├── requirements.txt
├── config.json
├── evaluate.py
├── inference.py
├── train.py
├── chkpt/trained_baseline_model/  # Need to Download
├── datasets/
│   ├── xacle_baseline_dataset.py
│   └── XACLE_dataset/              # Need to Download
├── losses/loss_function.py
├── models/
│   ├── Byola.py
│   ├── Roberta.py
│   ├── xacle_baseline_model.py
│   └── byola/chkpt/AudioNTT2022-BYOLA-64x96d2048.pth
├── pics/
└── utils/utils.py
```

<h2 id="usage">🚀 Usage</h2>

### For training (When learning from scratch)
```bash
python train.py
```
- A directory named "chkpt" is created, and within it, subdirectories based on the time when the learning proguram was executed are created (e.g., 20250901_1200).
- The JSON file (config.json) containing the learning settings is copied to the created subdirectory.
- The best model is saved as "best_model.pt" in the subdirectory.
- Training logs can be viewed via standard output and are saved as "log.txt" in the subdirectory created upon training completion.
### For Inference
```bash
python inference.py <chkpt_subdir_name> <dataset_key>
```
- Perform inference using the trained model.
- Cmd-Line argument descriptions
  - `<chkpt_subdir_name>`: Subdirectory name created during learning program execution (where the learning model is saved) (e.g., 20250901_1200)
  - `<dataset_key>`: Specify which dataset to use for inference.　Enter either *validation or test. If no argument is provided, inference will be performed on the validation data by default.
- Inference results are saved as "inferece_result_for_<dataset_key>.csv" in the subdirectory (`<chkpt_subdir_name>`).
  - The inference results are stored with the audio file name as the column name and the prediction score as the column name.

<h2 id="evaluation-code">✔ Evaluation Code</h2>

### For evaluation of the score prediction results for the validation data
```bash
python evaluate.py <inference_csv_path> <ground_truth_csv_path> <save_dir>
```
- Cmd-Line argument descriptions
  - `<inference_csv_path>`: Path to the CSV file containing the inference results for the validation data.
  - `<ground_truth_csv_path>`: Path to the CSV file containing the ground-truth scores for the validation data in XACLE dataset (validation_average.csv). 
  - `<save_dir>`: Directory where the evaluation result will be saved (the output file name is fixed as evaluation_result.csv)
- Using the predicted scores and ground-truth scores for the validation data, it calculates SRCC, LCC, KTAU, and MSE.
  - *This program cannot be used for predicting scores on test data because ground-truth is required.
- The results for SRCC, LCC, KTAU, MSE, and the number of evaluation data are written to a file named evaluation_result.csv inside `<save_dir>`.

<h2 id="result">💯 Results of baseline model for validation data</h2>


|  | SRCC | LCC | KTAU | MSE |
| :--- | ---: | ---: | ---: | ---: |
| Baseline | 0.384 |  0.396 | 0.264 | 4.836 |


<h2 id="license">📄 License</h2>

This project is licensed under the **MIT License**.

You are free to use, modify, and distribute this software, with or without modifications, under the conditions of the MIT License. See the [LICENSE](./LICENSE) file for full licese text.

<h2 id="citation">📚 Citation</h2>
Under preparation...
<!-- ```bibtex
@hogehoge{xacle2026,
    title={Xacle Challenge},
    author={hogehoge},
    journal={hogehoge},
    year={hogehoge}
}
``` -->

<h2 id="contributors">🧑‍💻Contributors<h2>

- Riki Takizawa (Kyoto Sangyo University, Japan)
- Yusuke Kanamori (The University of Tokyo, Japan)
- Yuki Okamoto (The University of Tokyo, Japan)
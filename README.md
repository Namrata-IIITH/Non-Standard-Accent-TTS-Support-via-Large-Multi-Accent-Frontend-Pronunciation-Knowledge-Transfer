# Multi-Accent TTS Frontend via Pronunciation Knowledge Transfer

A multi-accent text-to-phoneme frontend that models pronunciation variations across **14 English accents**, with a focus on data efficiency and accent similarity transfer.

This repository contains an implementation, dataset preparation scripts, experiment configurations, and evaluation code for reproducing our project based on [Berger et al., Interspeech 2025](https://github.com/sunsiqitos/multi_accent_s2s_frontend).

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Experiments](#experiments)
3. [Installation](#installation)
4. [Setup](#setup)
5. [Dataset Preparation](#dataset-preparation)
6. [Training](#training)
7. [Inference](#inference)
8. [Evaluation](#evaluation)
9. [Results](#results)

---

## Project Overview

Many English accents lack sufficient expert-curated pronunciation data needed for developing high-quality TTS systems. We address this by training a multi-accent Seq2Seq frontend that maps written text to phoneme sequences across 14 accents.

**Objectives:**
- Reduce the pronunciation data required for low-resource accents
- Evaluate a 14-accent multi-accent baseline
- Study scaling of pronunciation knowledge transfer (199k → 1k → 500 sentences)
- Analyze the role of accent similarity in transfer learning

---

## Experiments

### Experiment 1: Baseline Evaluation
Compare the 14-accent model vs. a single-accent model (EDI) using:
- Seen word accuracy
- Unseen word accuracy
- Boundary accuracy

### Experiment 2: Data Scaling (1k)
Fine-tune the baseline model with 1k sentences per accent (with ABD1 source augmentation) and evaluate unseen word accuracy on HiFi-TTS EDI.

### Experiment 3: Accent Similarity Transfer
Using the 1k setup, compare:
- **High similarity transfer:** ABD1 → EDI
- **Low similarity transfer:** GNZ → EDI

Measures how accent similarity affects prediction accuracy.

---

## Installation

**Create and activate a Conda environment:**

```bash
conda create -n multiaccent python=3.9
conda activate multiaccent
pip install -r requirements.txt
```

**Install Festival + Unilex:**

Follow the installation steps for Festival TTS and ensure the lexicons are placed in:
```
festival/lib/dicts/unilex/*.scm
```

---

## Setup

### 1. Clone the Base Repository

```bash
git clone https://github.com/sunsiqitos/multi_accent_s2s_frontend
cd multi_accent_s2s_frontend
```

### 2. Add Configuration Files

Place the following files into their respective folders:

```
config/edi/   ← edi_uni.json
config/multi/ ← multi_14_accents.json
```

### 3. Replace the Dataset Folder

Download the dataset from Hugging Face and replace the existing `dataset/` folder:

> 📦 [Multi-accent\_data on Hugging Face](https://huggingface.co/datasets/Jyo08/Multi-accent_data/tree/main)

### 4. Add Checkpoint and Output Folders

From the Hugging Face link, copy the following folders into the root of the cloned repository:

```
checkpoints/
checkpoints_uni/
uni_output/
results_multi/
```

### 5. Add Supporting Scripts

Place the following scripts in the root directory:

| Script | Description |
|---|---|
| `score_uni.py` | Computes scores (seen, unseen, boundary) for the EDI uni-accent model |
| `score.py` | Computes scores for the multi-accent model |
| `generate_config.py` | Configuration generator for the EDI model |
| `1k_data.py` | Randomly samples 1k sentences from the dataset |

---

## Dataset Preparation

### Step 1: Build Accent Lexicons

```bash
python data_scripts/build_lexicons.py
```

Converts all 14 Unisyn accent outputs into Festival `.scm` lexicons.

### Step 2: Bootstrap Text–Phoneme Files

```bash
python data_scripts/bootstrap_dataset.py
```

Generates the following splits for each accent (OOV sentences are removed to ensure phoneme consistency):

```
unilex_<accent>/src-train.txt    unilex_<accent>/tgt-train.txt
unilex_<accent>/src-val.txt      unilex_<accent>/tgt-val.txt
unilex_<accent>/src-test.txt     unilex_<accent>/tgt-test.txt
```

---

## Training

### A. Full 14-Accent Baseline (~199k sentences/accent)

```bash
python training/train.py \
    --config_path configs/multi_14accent.json \
    --output_path checkpoints/multi14
```

### B. Fine-Tuning for 1k Experiments

First generate reduced datasets using `1k_data.py`, then run:

```bash
python training/fine_tune.py \
    --config_path configs/multi_14accent_1k.json \
    --checkpoint checkpoints/multi14/step_x_epoch_y.pth.tar \
    --output_path checkpoints/multi14_1k
```

---

## Inference

```bash
python inference/inference.py \
    --config_fpath checkpoints/multi14/config.json \
    --src_vocab_fpath checkpoints/multi14/vocab/src.vocab \
    --tgt_vocab_fpath checkpoints/multi14/vocab/tgt.vocab \
    --restore_fpath checkpoints/multi14/step_x_epoch_y.pth.tar \
    --text_fpath datasets/unilex_edi/src-test.txt \
    --output_dir results/predictions \
    --lang <accent_index> \
    --decoding_method greedy
```

---

## Evaluation

### EDI (Uni-accent) Model

```bash
python score_uni.py \
    --train_src dataset/unilex_edi/src-train.txt \
    --test_src dataset/unilex_edi/src-test.txt \
    --test_tgt dataset/unilex_edi/tgt-test.txt \
    --pred_tgt uni_output/uni_edi/edi/src-test.ph.txt
```

### Multi-accent Model

```bash
python score.py \
    --train_src dataset/unilex_edi/src-train.txt \
    --test_src dataset/unilex_edi/src-test.txt \
    --test_tgt dataset/unilex_edi/tgt-test.txt \
    --pred_tgt results_multi/<run>/src-test.ph.txt
```

Both scripts compute:
- **Seen Word Accuracy**
- **Unseen Word Accuracy**
- **Boundary Accuracy**

---

## Results

Results tables and plots are available in `results/`, including:

```
results/
├── similarity_heatmap.png
├── baseline_results.txt
├── exp1k_abd1_edi_results.txt
└── exp1k_gnz_edi_results.txt
```

---

## Citation

If you use this work, please cite the original paper:

> Berger et al., *Interspeech 2025*

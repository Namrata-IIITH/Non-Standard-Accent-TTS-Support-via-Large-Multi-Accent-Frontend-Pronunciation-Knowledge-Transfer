# Non-Standard-Accent-TTS-Support-via-Large-Multi-Accent-Frontend-Pronunciation-Knowledge-Transfer
A multi-accent text-to-phoneme frontend that models pronunciation variations across 14 English accents, with a focus on data efficiency and accent similarity transfer.

This repository contains an implementation, dataset preparation scripts, experiment configurations, and evaluation code for reproducing our project based on Berger et al., Interspeech 2025.
The goal is to build a multi-accent text-to-phoneme frontend capable of modeling pronunciation variations across 14 English accents, with a focus on data efficiency and accent similarity transfer.

1. **Project Overview**

Many English accents lack sufficient expert-curated pronunciation data needed for developing high-quality TTS systems.
We address this problem by training a multi-accent Seq2Seq frontend, which maps written text to phoneme sequences across 14 accents.

Our objectives:

Reduce pronunciation data required for low-resource accents

Evaluate a 14-accent multi-accent baseline

Study scaling of pronunciation knowledge transfer (199k → 1k → 500)

Analyze role of accent similarity in transfer learning

**Experiments Implemented**
**Experiment 1: Baseline Evaluation**
Compare 14-accent model vs single-accent model (EDI) using:
1. Seen word accuracy
2. Unseen word accuracy
3. Boundary accuracy

**Experiment 2: Data Scaling (1k)**
Fine-tune baseline model with:
1. 1k sentences per accent (with ABD1 source augmentation)
2. Evaluate unseen word accuracy on HiFi-TTS EDI.

**Experiment 3: Accent Similarity Transfer**
Using 1k setup:
1. High similarity: ABD1 → EDI
2. Low similarity: GNZ → EDI
Measure how accent similarity affects prediction accuracy.

**Steps to run on local machine:**
**1. Clone the Repository**

Clone the project repository:
https://github.com/sunsiqitos/multi_accent_s2s_frontend

**2. Add Configuration Files**
Place the following files into the respective folders inside the cloned repository:
edi_uni.json → config/edi/
multi_14_accents.json → config/multi/

**3. Replace the Dataset Folder**
Replace the repository’s existing dataset folder with the dataset folder provided in the Hugging Face link:
(link)

**4. Add Required Output and Checkpoint Folders**
From the provided link, copy these folders into the cloned repository:
checkpoints
checkpoints_uni
uni_output
results_multi

(link)

**5. Add Supporting Python Files**

Place the following scripts in the root directory of the cloned repository:
score_uni.py – calculates scores for seen, unseen, and boundary words for the EDI accent model
score.py – calculates scores for seen, unseen, and boundary words for the multi-accent model
generate_config.py – configuration generator for the EDI model
1k_data.py – randomly samples 1k sentences from the dataset

**6. Training and Inference**
Training and inference commands are already provided in the repository’s README.
Use the same commands with the updated paths from your setup.

**7. Evaluation Commands**

a. **EDI (Uni-accent) Model**

python3 score_uni.py \
  --train_src dataset/unilex_edi/src-train.txt \
  --test_src dataset/unilex_edi/src-test.txt \
  --test_tgt dataset/unilex_edi/tgt-test.txt \
  --pred_tgt uni_output/uni_edi/edi/src-test.ph.txt

b. **Multi-accent Model**

python3 score.py \
  --train_src dataset/unilex_edi/src-train.txt \
  --test_src dataset/unilex_edi/src-test.txt \
  --test_tgt dataset/unilex_edi/tgt-test.txt \
  --pred_tgt results_multi/....../src-test.ph.txt

**Installation & Environment Setup**
Create Conda Environment:
conda create -n multiaccent python=3.9
conda activate multiaccent
pip install -r requirements.txt

**Install Festival + Unilex**

Follow installation steps for Festival TTS and ensure the lexicons are placed in:
festival/lib/dicts/unilex/*.scm

**Preparing the Pronunciation Dataset**
**Step 1: Build Accent Lexicons**
python data_scripts/build_lexicons.py
This converts all 14 Unisyn accent outputs into Festival .scm lexicons.

**Step 2: Bootstrap src/tgt Text–Phoneme Files**
python data_scripts/bootstrap_dataset.py

This generates:
unilex_<accent>/src-train.txt
unilex_<accent>/tgt-train.txt
unilex_<accent>/src-val.txt
unilex_<accent>/tgt-val.txt
unilex_<accent>/src-test.txt
unilex_<accent>/tgt-test.txt

All OOV sentences are removed to ensure phoneme consistency.

**Training the Models**
**A. Full 14-Accent Baseline (Ceiling Baseline)**
python training/train.py \
    --config_path configs/multi_14accent.json \
    --output_path checkpoints/multi14

This model is trained on ≈199k sentences per accent.

**B. Fine-Tuning for 1k Experiments**

First, generate reduced datasets (1k or 500 lines sampled from original training data).
Then run:

python training/fine_tune.py \
    --config_path configs/multi_14accent_1k.json \
    --checkpoint checkpoints/multi14/step_x_epoch_y.pth.tar \
    --output_path checkpoints/multi14_1k

**Inference**
python inference/inference.py \
    --config_fpath checkpoints/multi14/config.json \
    --src_vocab_fpath checkpoints/multi14/vocab/src.vocab \
    --tgt_vocab_fpath checkpoints/multi14/vocab/tgt.vocab \
    --restore_fpath checkpoints/multi14/step_x_epoch_y.pth.tar \
    --text_fpath datasets/unilex_edi/src-test.txt \
    --output_dir results/predictions \
    --lang <accent_index> \
    --decoding_method greedy

**Evaluation: Unseen Word Accuracy**
python inference/compute_metrics.py \
    --gold datasets/unilex_edi/tgt-test.txt \
    --pred results/predictions/pred.txt \
    --output results/metrics_edi.json

This computes:
Seen Word Accuracy
Unseen Word Accuracy
Boundary Accuracy

**Results**
Results tables and plots are provided in:

results/

Including:
similarity_heatmap.png
baseline_results.txt
exp1k_abd1_edi_results.txt
exp1k_gnz_edi_results.txt

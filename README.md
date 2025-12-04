# Non-Standard-Accent-TTS-Support-via-Large-Multi-Accent-Frontend-Pronunciation-Knowledge-Transfer
A multi-accent text-to-phoneme frontend that models pronunciation variations across 14 English accents, with a focus on data efficiency and accent similarity transfer.

This repository contains the full implementation, dataset preparation scripts, experiment configurations, and evaluation code for reproducing our project based on Berger et al., Interspeech 2025.
The goal is to build a multi-accent text-to-phoneme frontend capable of modeling pronunciation variations across 14 English accents, with a focus on data efficiency and accent similarity transfer.

1. **Project Overview**

Many English accents lack sufficient expert-curated pronunciation data needed for developing high-quality TTS systems.
We address this problem by training a multi-accent Seq2Seq frontend, which maps written text to phoneme sequences across 14 accents.

Our objectives:

Reduce pronunciation data required for low-resource accents

Evaluate a 14-accent multi-accent baseline

Study scaling of pronunciation knowledge transfer (199k → 1k → 500)

Analyze role of accent similarity in transfer learning

2. **Installation & Environment Setup**
Create Conda Environment:
conda create -n multiaccent python=3.9
conda activate multiaccent
pip install -r requirements.txt

**Install Festival + Unilex**

Follow installation steps for Festival TTS and ensure the lexicons are placed in:

festival/lib/dicts/unilex/*.scm

4. **Preparing the Pronunciation Dataset**
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

5. **Accent Similarity Analysis**

Compute the Levenshtein similarity matrix for all 14 accents:

python data_scripts/compute_similarity.py

Generates:
results/similarity_heatmap.png

6. **Training the Models**
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

7. **Inference**
python inference/inference.py \
    --config_fpath checkpoints/multi14/config.json \
    --src_vocab_fpath checkpoints/multi14/vocab/src.vocab \
    --tgt_vocab_fpath checkpoints/multi14/vocab/tgt.vocab \
    --restore_fpath checkpoints/multi14/step_x_epoch_y.pth.tar \
    --text_fpath datasets/unilex_edi/src-test.txt \
    --output_dir results/predictions \
    --lang <accent_index> \
    --decoding_method greedy

8. **Evaluation: Unseen Word Accuracy**
python inference/compute_metrics.py \
    --gold datasets/unilex_edi/tgt-test.txt \
    --pred results/predictions/pred.txt \
    --output results/metrics_edi.json

This computes:

Seen Word Accuracy

Unseen Word Accuracy

Boundary Accuracy

9. **Experiments Implemented**
**Experiment 1: Baseline Evaluation**

Compare 14-accent model vs single-accent model (EDI) using:

Seen word accuracy

Unseen word accuracy

Boundary accuracy

**Experiment 2: Data Scaling (1k)**

Fine-tune baseline model with:

1k sentences per accent (with ABD1 source augmentation)

Evaluate unseen word accuracy on HiFi-TTS EDI.

**Experiment 3: Accent Similarity Transfer**

Using 1k setup:

High similarity: ABD1 → EDI

Low similarity: GNZ → EDI

Measure how accent similarity affects prediction accuracy.

10. **Results**

Results tables and plots are provided in:

results/

Including:

similarity_heatmap.png

baseline_results.txt

exp1k_abd1_edi_results.txt

exp1k_gnz_edi_results.txt

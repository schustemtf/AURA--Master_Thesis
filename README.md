# AURA: Adaptive, Understandable and Reliable AI for KTx Aftercare

This repository extends [TFN (Temporal Fusion Nexus)](https://github.com/intelligentembeddedsystemslab/TFN_Temporal-Fusion-Nexus) with explainability techniques, uncertainty quantification, and an expanded expert study.

## Setup & Training Pipeline

To run any notebook via CLI (e.g. on a GPU server), use the provided `execute.sh` script.

### 1. Environment
Create and activate a Python virtual environment and install dependencies:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Prerequisites
- Create a [HuggingFace account](https://huggingface.co) and generate an access token
- Log into the HuggingFace CLI (required to download BERT):
```bash
huggingface-cli login
```
- Deploy NephroCAGE in `data/v1/`
  (*Note*: Not publicly available!)

### 3. Training

> **Note:** Steps 3.1–3.3 must be run in order, as each produces outputs required by the next.

#### 3.1 Fine-tune Language Model
Run `finetune_gte.ipynb` to fine-tune the BERT-based language model.  
**Output:** `models/med_gte.*`

#### 3.2 Generate Note Embeddings
Run `create_note_embeddings.ipynb` to embed clinical notes using the fine-tuned model.  
**Output:** `data/embeddings/emb_med_gte.*`

#### 3.3 Train T-LSTM
Run `training.ipynb` to train the T-LSTM for 10 epochs.  
**Output:** `models/after10epochs.pt`

#### 3.4 Train Classifier (MLP)
Run `classification.ipynb` to train the MLP (`clf_model`) on top of T-LSTM predictions,
mapping them to the final task prediction (e.g. GraftLoss).  
**Output:** `models/{task}@{days}_clf.pth`

### 4. Assembling the Full TFN Model
With all three components trained, they can be combined into the full TFN model.
See cells 1–5 of `interpret.ipynb` for a working example.

---

## Reproducing Results

### 1. Local Explainability
- `interpret-ig.ipynb` — computes IG or TIMING attributions and saves them to a `.json` file; all parameters are configurable within the notebook
- `local-results.ipynb` — plots and ranks attribution results, with SHAP comparison

### 2. Global Explainability
- `interpret-global.ipynb` — provides a slightly modified model architecture compatible with the [`zennit-crp`](https://github.com/rachtibat/zennit-crp) library, enabling CRP-based global explanations on TFN

### 3. Uncertainty Quantification
- `uncertainty.ipynb` — computes per-feature confidence estimates via ACI and RCPS, then compares and plots the results

### 4. User Study
- `new_study.ipynb` — processes user study results; running the notebook produces dataframes and plots of clinician feature rankings across all three tasks

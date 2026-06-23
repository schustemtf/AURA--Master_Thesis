#!/bin/sh
# ================================================================
# Run all training notebooks in order, waiting for each to finish 
# before the next. 
# Usage:
#     sh run_all.sh
# ================================================================

set -e  # abort immediately if any notebook fails

sh execute.sh finetune_gte.ipynb
sh execute.sh create_note_embeddings.ipynb
sh execute.sh training.ipynb
sh execute.sh classification.ipynb
sh execute.sh interpret.ipynb
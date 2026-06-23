#!/bin/bash
# ================================================================
# Run a Jupyter notebook headlessly (cluster-safe, detaches automatically)
# Extracts all text + HTML outputs into a timestamped log file.
# Usage:
#     sh run_notebook.sh notebook.ipynb
# ================================================================

if [ $# -lt 1 ]; then
  echo "Usage: sh $0 <notebook.ipynb>"
  exit 1
fi

NOTEBOOK="$1"
if [ ! -f "$NOTEBOOK" ]; then
  echo "Error: File '$NOTEBOOK' not found."
  exit 1
fi

LOG_DIR="./logs"
mkdir -p "$LOG_DIR"

BASENAME=$(basename "$NOTEBOOK" .ipynb)
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RAW_NOTEBOOK="executed_${BASENAME}_${TIMESTAMP}.ipynb"
LOGFILE="${LOG_DIR}/${BASENAME}_${TIMESTAMP}.log"
NOHUP_LOG="${LOG_DIR}/${BASENAME}_${TIMESTAMP}_nohup.log"

# Helper for dual printing (screen + log)
log() {
  echo "$@" | tee -a "$LOGFILE"
}

log "----------------------------------------------------------"
log " Running notebook:  $NOTEBOOK"
log " Temporary output:  $RAW_NOTEBOOK"
log " Log file:          $LOGFILE"
log " Started at:        $(date)"
log "----------------------------------------------------------"

# Run everything inside nohup bash -c block
nohup bash -c "
  echo '[$(date)] Executing notebook: $NOTEBOOK' >> '$LOGFILE'

  jupyter execute '$NOTEBOOK' >> '$LOGFILE' 2>&1

  awk '
    /\"text\": \[/ || /\"text\/plain\": \[/ || /\"text\/html\": \[/ {capture=1; next}
    capture {
      if (\$0 ~ /\]/) {capture=0}
      else {gsub(/[\"\\[\\],]/,\"\"); print}
    }
  ' '$NOTEBOOK' >> '$LOGFILE'

  echo '----------------------------------------------------------' >> '$LOGFILE'
  echo \" Extraction complete!\" >> '$LOGFILE'
  echo \" Finished at: \$(date)\" >> '$LOGFILE'
  echo \" Cleaned outputs saved to: $LOGFILE\" >> '$LOGFILE'
  echo '----------------------------------------------------------' >> '$LOGFILE'
" > "$NOHUP_LOG" 2>&1 &

PID=$!

log "Process started with PID: $PID"
log "To monitor progress: tail -f $LOGFILE"
log "----------------------------------------------------------"


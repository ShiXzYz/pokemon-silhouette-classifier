#!/usr/bin/env bash
# run_all.sh - Run the full pipeline in order
# Project: DS 4002 Project 3, "Who's That Pokemon?" (pokemon-silhouette-classifier)
# Authors: Brian Ryu, Keegan Jewell, Nathan Nguyen
#
# Purpose: data -> EDA -> k-NN -> CNN training + selection -> evaluation ->
#          Grad-CAM, timing each step and logging all output.
# Inputs:  internet on first run (sprites clone, ImageNet weights)
# Outputs: DATA/images/, MODEL/, OUTPUT/; log in OUTPUT/run_all_log.txt
# Run:     bash SCRIPTS/run_all.sh   (PYTHON=/path/to/python to pick interpreter)
# Runtime: ~15 min on Apple M2 Max (MPS); ~1-2 h on CPU

# Stop on any error
set -euo pipefail

# Run from repo root
cd "$(dirname "$0")/.."
PYTHON="${PYTHON:-python}"
LOG="OUTPUT/run_all_log.txt"
mkdir -p OUTPUT
: > "$LOG"   # fresh log

# Run a command, log its output, print its time
step() {
    local desc="$1"; shift
    echo "" | tee -a "$LOG"
    echo "=== [$(date '+%H:%M:%S')] $desc: $* ===" | tee -a "$LOG"
    local t0=$SECONDS
    "$@" 2>&1 | tee -a "$LOG"
    echo "--- $desc finished in $(( (SECONDS - t0) / 60 )) min $(( (SECONDS - t0) % 60 )) s" \
        | tee -a "$LOG"
}

# Record environment
step "Environment" "$PYTHON" -c "import sys, platform, torch; \
print('python', sys.version.split()[0], '|', platform.platform()); \
print('torch', torch.__version__, '| cuda', torch.cuda.is_available(), \
'| mps', torch.backends.mps.is_available())"

step "00 Build dataset"     "$PYTHON" SCRIPTS/00_build_dataset.py
step "01 EDA figures"       "$PYTHON" SCRIPTS/01_eda_figures.py
step "02 k-NN baseline"     "$PYTHON" SCRIPTS/02_knn_baseline.py

# Fresh validation logs
rm -f OUTPUT/tables/val_runs.csv OUTPUT/tables/training_history.csv
step "03 Train color, with Gen I"         "$PYTHON" SCRIPTS/03_train_cnn.py --mode color
step "03 Train color, without Gen I"      "$PYTHON" SCRIPTS/03_train_cnn.py --mode color --exclude-gen1
step "03 Train silhouette, with Gen I"    "$PYTHON" SCRIPTS/03_train_cnn.py --mode silhouette
step "03 Train silhouette, without Gen I" "$PYTHON" SCRIPTS/03_train_cnn.py --mode silhouette --exclude-gen1
step "03 Select final models"             "$PYTHON" SCRIPTS/03_train_cnn.py --select

step "04 Evaluate on HOME"  "$PYTHON" SCRIPTS/04_evaluate.py
step "05 Grad-CAM"          "$PYTHON" SCRIPTS/05_gradcam.py

echo "" | tee -a "$LOG"
echo "All steps finished in $(( SECONDS / 60 )) min. Log: $LOG" | tee -a "$LOG"

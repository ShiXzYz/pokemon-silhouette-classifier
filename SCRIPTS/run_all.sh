#!/usr/bin/env bash
# =============================================================================
# run_all.sh - Reproduce the whole project, start to finish
# =============================================================================
# Project : DS 4002 Project 3, "Who's That Pokemon?" (pokemon-silhouette-classifier)
# Authors : Brian Ryu, Keegan Jewell, Nathan Nguyen
#
# PURPOSE
#     Runs every script in order: build data -> EDA -> k-NN baseline ->
#     12 CNN training runs + model selection -> test evaluation -> Grad-CAM.
#     Prints the wall-clock time of each step and saves a full log.
#
# INPUTS
#     Internet access (first run only: PokeAPI sprites clone + ImageNet weights).
#
# OUTPUTS
#     Everything in DATA/images/, models/, OUTPUT/figures/, OUTPUT/tables/,
#     plus the console log in OUTPUT/run_all_log.txt.
#
# HOW TO RUN (from the repository root, with the environment activated)
#     bash SCRIPTS/run_all.sh
#     PYTHON=/path/to/python bash SCRIPTS/run_all.sh   # to choose the interpreter
#
# EXPECTED RUNTIME
#     ~15 min on an Apple M2 Max (MPS GPU); 14 of those are 03_train_cnn.py.
#     Roughly 1-2 h on CPU only.
# =============================================================================

# Stop at the first failing command (including failures inside a pipe).
set -euo pipefail

# Always run from the repo root, whatever directory the script was called from.
cd "$(dirname "$0")/.."
PYTHON="${PYTHON:-python}"
LOG="OUTPUT/run_all_log.txt"
mkdir -p OUTPUT
: > "$LOG"   # start a fresh log

# step "<description>" <command...>: run a command, echo + log its output, time it.
step() {
    local desc="$1"; shift
    echo "" | tee -a "$LOG"
    echo "=== [$(date '+%H:%M:%S')] $desc: $* ===" | tee -a "$LOG"
    local t0=$SECONDS
    "$@" 2>&1 | tee -a "$LOG"
    echo "--- $desc finished in $(( (SECONDS - t0) / 60 )) min $(( (SECONDS - t0) % 60 )) s" \
        | tee -a "$LOG"
}

# Log the software environment for the record.
step "Environment" "$PYTHON" -c "import sys, platform, torch; \
print('python', sys.version.split()[0], '|', platform.platform()); \
print('torch', torch.__version__, '| cuda', torch.cuda.is_available(), \
'| mps', torch.backends.mps.is_available())"

step "00 Build dataset"     "$PYTHON" SCRIPTS/00_build_dataset.py
step "01 EDA figures"       "$PYTHON" SCRIPTS/01_eda_figures.py
step "02 k-NN baseline"     "$PYTHON" SCRIPTS/02_knn_baseline.py

# Start the validation log from scratch so it only holds this run's 12 rows.
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

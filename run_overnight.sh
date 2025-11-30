#!/bin/bash
# Overnight TDA Batch Processing
# 
# This script runs TDA analysis in the background with logging.
# Output is saved to a log file so you can monitor progress.
#
# Usage:
#   ./run_overnight.sh perturbation              # Run perturbation TDA (09_)
#   ./run_overnight.sh perturbation --top 10000  # Perturbation with top 10k
#   ./run_overnight.sh bifiltration              # Run bifiltration TDA (18_)
#   ./run_overnight.sh bifiltration --ad-only    # Bifiltration AD only
#   ./run_overnight.sh candidates                # Run bifiltration on candidates (24_)
#   ./run_overnight.sh candidates --projects     # Bifiltration for all disease projects

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Determine which script to run
MODE=$1
shift  # Remove first argument, pass rest to python script

if [ "$MODE" = "bifiltration" ] || [ "$MODE" = "bifilt" ] || [ "$MODE" = "18" ]; then
    SCRIPT="18_overnight_bifiltration.py"
    LOG_FILE="overnight_bifiltration_${TIMESTAMP}.txt"
    PID_FILE="overnight_bifiltration.pid"
    echo "Running BIFILTRATION TDA..."
elif [ "$MODE" = "perturbation" ] || [ "$MODE" = "perturb" ] || [ "$MODE" = "09" ]; then
    SCRIPT="09_overnight_batch_tda.py"
    LOG_FILE="overnight_perturbation_${TIMESTAMP}.txt"
    PID_FILE="overnight_perturbation.pid"
    echo "Running PERTURBATION TDA..."
elif [ "$MODE" = "candidates" ] || [ "$MODE" = "cand" ] || [ "$MODE" = "24" ]; then
    SCRIPT="24_overnight_bifiltration_candidates.py"
    LOG_FILE="overnight_candidates_${TIMESTAMP}.txt"
    PID_FILE="overnight_candidates.pid"
    echo "Running BIFILTRATION on CANDIDATES..."
else
    echo "Usage: ./run_overnight.sh <mode> [options]"
    echo ""
    echo "Modes:"
    echo "  perturbation (or perturb, 09)  - Run 09_overnight_batch_tda.py"
    echo "  bifiltration (or bifilt, 18)   - Run 18_overnight_bifiltration.py"
    echo "  candidates (or cand, 24)       - Run 24_overnight_bifiltration_candidates.py"
    echo ""
    echo "Examples:"
    echo "  ./run_overnight.sh perturbation --projects"
    echo "  ./run_overnight.sh perturbation --top 10000"
    echo "  ./run_overnight.sh bifiltration"
    echo "  ./run_overnight.sh bifiltration --ad-only"
    echo "  ./run_overnight.sh candidates --projects --candidates"
    echo "  ./run_overnight.sh candidates --n-candidates 5000"
    exit 1
fi

echo "Script: $SCRIPT"
echo "Log file: $LOG_FILE"
echo "Arguments: $@"
echo ""
echo "To monitor progress:"
echo "  tail -f $LOG_FILE"
echo ""
echo "To stop the process:"
echo "  kill \$(cat $PID_FILE)"
echo ""

# Run in background with nohup so it survives terminal closing
# -u flag forces unbuffered output for real-time logging
nohup python -u "$SCRIPT" "$@" > "$LOG_FILE" 2>&1 &
echo $! > "$PID_FILE"

echo "Process started with PID: $(cat $PID_FILE)"
echo "Process will continue even if you close this terminal."
echo ""
echo "Quick commands:"
echo "  Check progress:  tail -20 $LOG_FILE"
echo "  Live monitor:    tail -f $LOG_FILE"
echo "  Stop process:    kill \$(cat $PID_FILE)"


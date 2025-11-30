#!/bin/bash
# Overnight TDA Batch Processing
# 
# This script runs the TDA analysis in the background with logging.
# Output is saved to overnight_log.txt so you can monitor progress.
#
# Usage:
#   ./run_overnight.sh                    # Run all projects
#   ./run_overnight.sh --top 10000        # Run top 10000 candidates
#   ./run_overnight.sh --projects --top 5000  # Run both

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="overnight_log_${TIMESTAMP}.txt"

echo "Starting overnight TDA batch processing..."
echo "Log file: $LOG_FILE"
echo "Arguments: $@"
echo ""
echo "To monitor progress:"
echo "  tail -f $LOG_FILE"
echo ""
echo "To stop the process:"
echo "  kill \$(cat overnight.pid)"
echo ""

# Run in background with nohup so it survives terminal closing
# -u flag forces unbuffered output for real-time logging
nohup python -u 09_overnight_batch_tda.py "$@" > "$LOG_FILE" 2>&1 &
echo $! > overnight.pid

echo "Process started with PID: $(cat overnight.pid)"
echo "Process will continue even if you close this terminal."
echo ""
echo "Quick commands:"
echo "  Check progress:  tail -20 $LOG_FILE"
echo "  Live monitor:    tail -f $LOG_FILE"
echo "  Stop process:    kill \$(cat overnight.pid)"


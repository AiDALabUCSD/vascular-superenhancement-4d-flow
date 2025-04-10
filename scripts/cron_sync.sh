#!/bin/bash

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Set up logging
LOG_DIR="$PROJECT_ROOT/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/sync.log"

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate /mnt/yeluru/.local/share/mamba/envs/vascular-superenhancement-4d-flow 2>> "$LOG_FILE"

# Create a temporary file for the Python script output
TEMP_OUTPUT=$(mktemp)

# Run the sync script and capture output
python "$SCRIPT_DIR/sync_to_nas.py" > "$TEMP_OUTPUT" 2>&1
EXIT_CODE=$?

# Only log if there was output (indicating changes) or an error
if [ -s "$TEMP_OUTPUT" ] || [ $EXIT_CODE -ne 0 ]; then
    echo "=== Starting sync at $(date) ===" >> "$LOG_FILE"
    cat "$TEMP_OUTPUT" >> "$LOG_FILE"
    if [ $EXIT_CODE -eq 0 ]; then
        echo "Sync completed successfully at $(date)" >> "$LOG_FILE"
    else
        echo "Sync failed with exit code $EXIT_CODE at $(date)" >> "$LOG_FILE"
    fi
    echo "=== End of sync operation ===" >> "$LOG_FILE"
    echo "" >> "$LOG_FILE"
fi

# Clean up temporary file
rm "$TEMP_OUTPUT"

exit $EXIT_CODE 
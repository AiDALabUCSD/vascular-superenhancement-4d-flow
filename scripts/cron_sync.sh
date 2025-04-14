#!/bin/bash

# This script runs weekly to sync working directory to NAS
# For manual backups, create a file named 'trigger_backup' in the project root

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Check for manual backup trigger
TRIGGER_FILE="$PROJECT_ROOT/trigger_backup"
if [ ! -f "$TRIGGER_FILE" ]; then
    # If no trigger file exists, check if it's the right day for weekly backup
    # Run only on Sundays at 2 AM
    if [ "$(date +%u)" != "7" ] || [ "$(date +%H)" != "02" ]; then
        exit 0
    fi
fi

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

# Handle output and trigger file
if [ $EXIT_CODE -eq 0 ]; then
    # If there was output (changes) or this was a manual backup, log it
    if [ -s "$TEMP_OUTPUT" ] || [ -f "$TRIGGER_FILE" ]; then
        echo "=== Starting sync at $(date) ===" >> "$LOG_FILE"
        if [ -s "$TEMP_OUTPUT" ]; then
            cat "$TEMP_OUTPUT" >> "$LOG_FILE"
            echo "Sync completed successfully at $(date)" >> "$LOG_FILE"
        else
            echo "No changes to sync at $(date)" >> "$LOG_FILE"
        fi
        echo "=== End of sync operation ===" >> "$LOG_FILE"
        echo "" >> "$LOG_FILE"
    fi
    
    # Remove trigger file if this was a manual backup
    if [ -f "$TRIGGER_FILE" ]; then
        rm "$TRIGGER_FILE"
        echo "Manual backup trigger file removed at $(date)" >> "$LOG_FILE"
    fi
else
    # Log error
    echo "=== Starting sync at $(date) ===" >> "$LOG_FILE"
    cat "$TEMP_OUTPUT" >> "$LOG_FILE"
    echo "Sync failed with exit code $EXIT_CODE at $(date)" >> "$LOG_FILE"
    echo "=== End of sync operation ===" >> "$LOG_FILE"
    echo "" >> "$LOG_FILE"
fi

# Clean up temporary file
rm "$TEMP_OUTPUT"

exit $EXIT_CODE 
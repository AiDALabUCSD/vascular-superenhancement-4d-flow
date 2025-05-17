#!/bin/bash

# This script runs weekly to sync working directory to NAS
# For manual backups, create a file named 'trigger_backup' in the project root

# Enable error handling
set -e
set -o pipefail

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Set up logging
LOG_DIR="$PROJECT_ROOT/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/sync.log"

echo "=== Starting backup process at $(date) ===" >> "$LOG_FILE"

# Check for manual backup trigger
TRIGGER_FILE="$PROJECT_ROOT/trigger_backup"
if [ ! -f "$TRIGGER_FILE" ]; then
    # If no trigger file exists, check if it's the right day for weekly backup
    # Run only on Sundays at 2 AM
    if [ "$(date +%u)" != "7" ] || [ "$(date +%H)" != "02" ]; then
        echo "Not time for scheduled backup, exiting" >> "$LOG_FILE"
        exit 0
    fi
fi

# Create a temporary file for the Python script output
TEMP_OUTPUT=$(mktemp)
echo "Created temporary output file: $TEMP_OUTPUT" >> "$LOG_FILE"

# Check if conda is available
if [ ! -f ~/miniconda3/etc/profile.d/conda.sh ]; then
    echo "Error: conda.sh not found at ~/miniconda3/etc/profile.d/conda.sh" >> "$LOG_FILE"
    exit 1
fi

# Activate conda environment and run the sync script
echo "Activating conda environment..." >> "$LOG_FILE"
source ~/miniconda3/etc/profile.d/conda.sh

# Check if the conda environment exists
CONDA_ENV="/mnt/yeluru/.local/share/mamba/envs/vascular-superenhancement-4d-flow"
if [ ! -d "$CONDA_ENV" ]; then
    echo "Error: Conda environment not found at $CONDA_ENV" >> "$LOG_FILE"
    exit 1
fi

echo "Running sync script..." >> "$LOG_FILE"

# Try to run the script with conda run
if ! conda run -p "$CONDA_ENV" python "$SCRIPT_DIR/sync_to_nas.py" > "$TEMP_OUTPUT" 2>&1; then
    echo "Failed to run with conda run, trying direct Python execution..." >> "$LOG_FILE"
    # If conda run fails, try running Python directly
    if ! python "$SCRIPT_DIR/sync_to_nas.py" > "$TEMP_OUTPUT" 2>&1; then
        echo "Both conda run and direct Python execution failed" >> "$LOG_FILE"
        cat "$TEMP_OUTPUT" >> "$LOG_FILE"
        rm "$TEMP_OUTPUT"
        exit 1
    fi
fi

EXIT_CODE=$?
echo "Sync script completed with exit code: $EXIT_CODE" >> "$LOG_FILE"

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
    echo "Sync script output:" >> "$LOG_FILE"
    cat "$TEMP_OUTPUT" >> "$LOG_FILE"
    echo "Sync failed with exit code $EXIT_CODE at $(date)" >> "$LOG_FILE"
    echo "=== End of sync operation ===" >> "$LOG_FILE"
    echo "" >> "$LOG_FILE"
fi

# Clean up temporary file
rm "$TEMP_OUTPUT"
echo "Temporary file removed" >> "$LOG_FILE"

echo "=== Backup process completed at $(date) ===" >> "$LOG_FILE"
exit $EXIT_CODE 
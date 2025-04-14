#!/bin/bash

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Create the trigger file
touch "$PROJECT_ROOT/trigger_backup"

# Run the backup script
"$SCRIPT_DIR/cron_sync.sh" 
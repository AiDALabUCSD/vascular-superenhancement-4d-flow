#!/bin/bash
# Mirror /home/ayeluru/vascular-superenhancement-4d-flow/working_dir/ to fourier
# over rsync-over-SSH (NOT a CIFS mount).
#
# Why SSH and not the SMB/CIFS mount at /home/ayeluru/mnt/fourier/...?
#   The CIFS kernel module accumulates unaccounted page-cache memory under
#   sustained slow writes (observed ~10 GB/hour growth). That eventually
#   triggers memory thrashing on mina and forces a hard reboot. Three confirmed
#   incidents in May 2026. rsync-over-SSH bypasses CIFS entirely; mina sees
#   only TCP back-pressure (bounded socket buffers), so memory stays flat.
#
# Usage:
#   ./scripts/sync_to_fourier_ssh.sh                  # real run, --bwlimit=100M (default)
#   ./scripts/sync_to_fourier_ssh.sh --dry-run        # preview only, no transfer
#   ./scripts/sync_to_fourier_ssh.sh --bwlimit=50M    # custom bandwidth cap
#   ./scripts/sync_to_fourier_ssh.sh --bwlimit=0      # uncapped (will saturate 1 GbE)
#
# Recommended bwlimit values on this 1 GbE link:
#   50M  -> ~5 hr for 873 GB, interactive sessions stay snappy
#   100M -> ~2.5 hr, mild lag on SSH/Cursor   (DEFAULT - good tradeoff)
#   0    -> ~2.4 hr, eno1 saturated, interactive feels laggy
#
# Run inside tmux from a regular SSH session so it survives disconnects:
#   tmux new-session -s sync
#   ./scripts/sync_to_fourier_ssh.sh
#   Ctrl-b d  (detach)
#   tmux attach -t sync  (reattach later)

set -u

# ---- configuration ---------------------------------------------------------
SRC="/home/ayeluru/vascular-superenhancement-4d-flow/working_dir/"
REMOTE_USER="ayeluru"
REMOTE_HOST="137.110.172.30"
REMOTE_PATH="/volume1/projects/vascular-superenhancement-4d-flow/"
DST="${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PATH}"

BWLIMIT_DEFAULT="100M"
SSH_CIPHER="aes128-gcm@openssh.com"

EXCLUDES=(
    --exclude='logs/'
    --exclude='wandb/'
    --exclude='**/dicom_predictions/'
    --exclude='*.dcm'
)

# ---- argument parsing ------------------------------------------------------
DRY_RUN=""
BWLIMIT="$BWLIMIT_DEFAULT"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run|-n)
            DRY_RUN="--dry-run"
            shift
            ;;
        --bwlimit=*)
            BWLIMIT="${1#--bwlimit=}"
            shift
            ;;
        -h|--help)
            sed -n '2,30p' "$0"
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            echo "Use --help for usage." >&2
            exit 1
            ;;
    esac
done

# ---- preflight -------------------------------------------------------------
if [[ ! -d "$SRC" ]]; then
    echo "ERROR: source does not exist: $SRC" >&2
    exit 1
fi

if ! ssh -o ConnectTimeout=10 -o BatchMode=yes \
        "${REMOTE_USER}@${REMOTE_HOST}" 'true' 2>/dev/null; then
    echo "ERROR: cannot reach ${REMOTE_HOST} over SSH with passwordless auth." >&2
    echo "Set up keys first:  ssh-copy-id ${REMOTE_USER}@${REMOTE_HOST}" >&2
    exit 1
fi

LOG="$HOME/rsync-fourier-ssh-$(date +%Y%m%d-%H%M%S).log"
echo "==========================================================="
echo "  Source:      $SRC"
echo "  Destination: $DST"
echo "  Bandwidth:   ${BWLIMIT} ${DRY_RUN:+(DRY RUN - no data will move)}"
echo "  Log file:    $LOG"
echo "  Started:     $(date -Is)"
echo "==========================================================="
echo

# ---- retry loop ------------------------------------------------------------
attempt=0
while :; do
    attempt=$((attempt + 1))
    echo "=== attempt #$attempt at $(date -Is) ===" | tee -a "$LOG"

    if ! ssh -o ConnectTimeout=10 -o BatchMode=yes \
            "${REMOTE_USER}@${REMOTE_HOST}" 'true' 2>>"$LOG"; then
        echo "  SSH unreachable, sleeping 60s..." | tee -a "$LOG"
        sleep 60
        continue
    fi

    rsync -ai $DRY_RUN \
        --info=progress2,stats2 \
        --no-compress -W \
        --partial --partial-dir=.rsync-partial \
        --bwlimit="$BWLIMIT" \
        "${EXCLUDES[@]}" \
        --no-owner --no-group --no-perms \
        --log-file="$LOG" \
        -e "ssh -c ${SSH_CIPHER} -o ServerAliveInterval=30 -o ServerAliveCountMax=10" \
        "$SRC" "$DST"
    rc=$?

    echo "=== rsync exited rc=$rc at $(date -Is) ===" | tee -a "$LOG"
    case $rc in
        0)
            echo "DONE."
            exit 0
            ;;
        20)
            echo "Interrupted by signal. Stopping."
            exit 20
            ;;
        255)
            echo "SSH-level error. Sleeping 30s before retry..."
            sleep 30
            ;;
        30|35)
            echo "Timeout (rc=$rc). Sleeping 30s before retry..."
            sleep 30
            ;;
        23|24)
            echo "Vanished files / partial perms error (rc=$rc). Sleeping 30s..."
            sleep 30
            ;;
        *)
            echo "Unexpected rc=$rc. Sleeping 60s before retry..."
            sleep 60
            ;;
    esac
done

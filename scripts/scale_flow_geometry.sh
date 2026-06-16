#!/usr/bin/env bash
# Scale the flow-measurement precompute across a split (default: validation).
#
# Resumable end-to-end chain:
#   1) prepare native auto-flow inputs from this project's NIfTIs (main env)
#   2) run auto-flow geometry chain on GPU (auto-flow env, skips already-done)
#   3) build the compact flow_geometry.npz caches (main env)
#
# Usage:
#   bash scripts/scale_flow_geometry.sh [SPLIT]
#   SPLIT defaults to "validation".
set -euo pipefail

SPLIT="${1:-validation}"
MAIN_ENV="vascular-superenhancement-4d-flow"
AUTOFLOW_ENV="auto-flow_3-9"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
MANIFEST="working_dir/all_patients/autoflow_manifest_${SPLIT}.csv"

echo "=== STEP 1: prepare native inputs [split=$SPLIT] ($(date)) ==="
conda run -n "$MAIN_ENV" --no-capture-output \
  python scripts/prepare_autoflow_inputs.py --split "$SPLIT" --manifest "$MANIFEST"

echo "=== STEP 2: geometry chain ($(date)) ==="
conda run -n "$AUTOFLOW_ENV" --no-capture-output \
  python scripts/run_autoflow_geometry.py --manifest "$MANIFEST" --skip-convert --skip-existing

echo "=== STEP 3: build flow geometry caches ($(date)) ==="
conda run -n "$MAIN_ENV" --no-capture-output \
  python scripts/build_flow_geometry_cache.py --split "$SPLIT"

echo "=== ALL DONE [split=$SPLIT] ($(date)) ==="

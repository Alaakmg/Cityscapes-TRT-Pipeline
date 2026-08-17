#!/usr/bin/env bash
# Bootstrap a fresh RunPod pod for training.
#
# Expects the network volume mounted at /workspace with the Cityscapes zips
# under /workspace/datasets/. Extracts them once (skipped if already done),
# clones/updates the repo, installs it, and prints a CUDA sanity check.
#
#   bash scripts/pod_bootstrap.sh
set -euo pipefail

WORKSPACE=/workspace
DATA_ROOT=$WORKSPACE/cityscapes
REPO_DIR=$WORKSPACE/cityscapes-trt-pipeline
REPO_URL=https://github.com/Alaakmg/Cityscapes-TRT-Pipeline.git

echo "== dataset =="
if [ -d "$DATA_ROOT/leftImg8bit/train" ] && [ -d "$DATA_ROOT/gtFine/train" ]; then
    echo "already extracted at $DATA_ROOT"
else
    mkdir -p "$DATA_ROOT"
    unzip -q -n "$WORKSPACE/datasets/leftImg8bit_trainvaltest.zip" -d "$DATA_ROOT"
    unzip -q -n "$WORKSPACE/datasets/gtFine_trainvaltest.zip" -d "$DATA_ROOT"
    echo "extracted to $DATA_ROOT"
fi
echo "train images: $(find "$DATA_ROOT/leftImg8bit/train" -name '*_leftImg8bit.png' | wc -l) (expect 2975)"
echo "val images:   $(find "$DATA_ROOT/leftImg8bit/val" -name '*_leftImg8bit.png' | wc -l) (expect 500)"

echo "== repo =="
if [ -d "$REPO_DIR/.git" ]; then
    git -C "$REPO_DIR" pull --ff-only
else
    git clone "$REPO_URL" "$REPO_DIR"
fi
cd "$REPO_DIR"
pip install -q -e ".[dev]"

echo "== sanity =="
python - <<'EOF'
import torch
assert torch.cuda.is_available(), "CUDA not available!"
print(f"torch {torch.__version__} | {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
EOF
pytest -q

echo
echo "ready. smoke test with:"
echo "  python -m segdeploy.train --config configs/runpod.yaml   # ctrl-c after ~1 epoch"

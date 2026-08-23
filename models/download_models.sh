#!/usr/bin/env bash
# Provisions the pretrained weights required by the pipeline into models/.
# Both files are gitignored (models/*.pt) or may not be present on a fresh
# checkout, so this script is idempotent: it only downloads what's missing.
#
# NOT handled here (see README "Data & Model Provisioning"):
#   data/videos/entrance.mp4, data/videos/interior.mp4 — source footage is
#   not redistributable and must be placed there manually.
set -e
cd "$(dirname "$0")"

download() {
    local file="$1"
    if [ -f "$file" ]; then
        echo "  models/$file already present, skipping."
        return
    fi
    echo "  Downloading models/$file ..."
    python -c "from ultralytics.utils.downloads import attempt_download_asset; attempt_download_asset('$file')"
}

echo "Provisioning pipeline models..."

# YOLO11m-Pose: 17-keypoint pose detector, used for both scenes (Pass 1 perception).
# See src/cli.py :: run_entrance / run_interior.
download "yolo11m-pose.pt"

# yolo26n-reid: BoT-SORT deep appearance ReID network, referenced by
# configs/botsort_entrance.yaml and configs/botsort_interior.yaml (model: key).
download "yolo26n-reid.onnx"

echo ""
echo "Model provisioning complete:"
ls -lh yolo11m-pose.pt yolo26n-reid.onnx

# Videos can't be fetched automatically -- warn if they're still missing.
videos_dir="../data/videos"
missing=0
for v in entrance.mp4 interior.mp4; do
    if [ ! -f "$videos_dir/$v" ]; then
        echo "WARNING: $videos_dir/$v not found. Place the source footage there before running the CLI."
        missing=1
    fi
done
if [ "$missing" -eq 0 ]; then
    echo "Source videos found in data/videos/. Ready to run the pipeline."
fi

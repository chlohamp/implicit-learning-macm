#!/bin/bash
#SBATCH --job-name=macm_array
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8gb
#SBATCH --account=iacc_nbc
#SBATCH --qos=pq_nbc
#SBATCH --partition=IB_44C_512G
#SBATCH --output=/home/data/nbc/misc-projects/meta-analyses/implicit-learning-macm/log/macm_array/%x_%A_%a.out
#SBATCH --error=/home/data/nbc/misc-projects/meta-analyses/implicit-learning-macm/log/macm_array/%x_%A_%a.err

pwd; hostname; date

# ====== Env ======
set +u
module add miniconda3-4.5.11-gcc-8.2.0-oqs2mbg
ENV_PATH="/home/data/nbc/misc-projects/meta-analyses/implicit-learning-macm/macm-env"
if [ -f "$(conda info --base 2>/dev/null)/etc/profile.d/conda.sh" ]; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "$ENV_PATH" || conda activate macm-env
else
  source activate "$ENV_PATH" 2>/dev/null || source activate macm-env
fi
set -euo pipefail

# ====== Paths ======
PROJECT_DIR="/home/data/nbc/misc-projects/meta-analyses/implicit-learning-macm"
ROI_DIR="${PROJECT_DIR}/dset/ROIs"
LIST="${PROJECT_DIR}/roi_list.txt"
LOG_DIR="${PROJECT_DIR}/log/macm_array"
mkdir -p "$LOG_DIR"

[ -d "$PROJECT_DIR" ] || { echo "Error: Project dir not found: $PROJECT_DIR"; exit 1; }
[ -d "$ROI_DIR" ] || { echo "Error: ROI dir not found: $ROI_DIR"; exit 1; }

# ====== Rebuild list (follow symlinks; sanitize CRLF; drop blanks/dups) ======
echo "Rebuilding ROI list from: $ROI_DIR"
find -L "$ROI_DIR" -maxdepth 1 -type f \( -iname "*.nii" -o -iname "*.nii.gz" \) -print0 \
| sort -z \
| tr '\0' '\n' \
| sed 's/\r$//' \
| awk 'NF' \
| awk '!seen[$0]++' \
> "$LIST"

if [ ! -s "$LIST" ]; then
  echo "Error: No ROI files found in $ROI_DIR"
  ls -la "$ROI_DIR"
  exit 1
fi

# Load into an array (one path per element)
mapfile -t ROIS < "$LIST"
NUM=${#ROIS[@]}
echo "Found $NUM ROI file(s)."

echo "---- zero-based index -> ROI path ----"
nl -ba -v 0 -w1 -s': ' "$LIST"
echo "--------------------------------------"

# Default SLURM_ARRAY_TASK_ID=0 if not set (allows single-task runs)
: "${SLURM_ARRAY_TASK_ID:=0}"
echo "Array task index: ${SLURM_ARRAY_TASK_ID}"

# Bounds check
if (( SLURM_ARRAY_TASK_ID < 0 || SLURM_ARRAY_TASK_ID >= NUM )); then
  echo "Array index ${SLURM_ARRAY_TASK_ID} out of range [0..$((NUM-1))]; exiting."
  exit 0
fi

ROI_FILE="${ROIS[$SLURM_ARRAY_TASK_ID]}"
echo "Selected ROI (index ${SLURM_ARRAY_TASK_ID}): $ROI_FILE"
[ -f "$ROI_FILE" ] || { echo "Error: ROI file not found: $ROI_FILE"; exit 1; }

# ====== Sanity check env ======
echo "Python: $(python --version)"
python - <<'PYCHK' || { echo "Error: NiMARE not available"; exit 1; }
import nimare
print("NiMARE:", nimare.__version__)
PYCHK

# ====== Run ======
CMD="python ${PROJECT_DIR}/macm_workflow_array.py \
  --project_dir ${PROJECT_DIR} \
  --roi_dir ${ROI_DIR} \
  --roi_file \"${ROI_FILE}\" \
  --n_cores ${SLURM_CPUS_PER_TASK}"

echo "Commandline: $CMD"
eval "$CMD"

echo "Done."
date

#!/bin/bash
#SBATCH --job-name=macm_array
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8gb
#SBATCH --account=iacc_nbc
#SBATCH --qos=pq_nbc
#SBATCH --partition=IB_44C_512G
# NOTE: pass the array range at submission time (see sbatch command below)
#SBATCH --output=/home/data/nbc/misc-projects/meta-analyses/implicit-learning-macm/log/macm_array/%x_%A_%a.out
#SBATCH --error=/home/data/nbc/misc-projects/meta-analyses/implicit-learning-macm/log/macm_array/%x_%A_%a.err

set -euo pipefail
pwd; hostname; date

# ====== Env ======
module add miniconda3-4.5.11-gcc-8.2.0-oqs2mbg
source /home/data/nbc/misc-projects/Hampson_Habenula/env/activate_env

# ====== Paths ======
PROJECT_DIR="/home/data/nbc/misc-projects/meta-analyses/implicit-learning-macm"
ROI_DIR="${PROJECT_DIR}/dset/ROIs"
CODE_DIR="${PROJECT_DIR}/code"
LIST="${PROJECT_DIR}/roi_list.txt"

# Build ROI list if missing
if [ ! -f "$LIST" ]; then
  { ls -1 "${ROI_DIR}"/*.nii "${ROI_DIR}"/*.nii.gz 2>/dev/null || true; } | sort > "$LIST"
fi

# Bound-check in case array range is larger than list
NUM=$(wc -l < "$LIST" | tr -d ' ')
if [ "${SLURM_ARRAY_TASK_ID}" -ge "${NUM}" ]; then
  echo "Array index ${SLURM_ARRAY_TASK_ID} >= ${NUM}; nothing to do. Exiting."
  exit 0
fi

ROI_FILE=$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" "$LIST")
echo "Selected ROI: $ROI_FILE"

cmd="python ${CODE_DIR}/macm_workflow_array.py \
  --project_dir ${PROJECT_DIR} \
  --roi_dir ${ROI_DIR} \
  --roi_file ${ROI_FILE} \
  --n_cores ${SLURM_CPUS_PER_TASK}"

echo "Commandline: $cmd"
eval $cmd

echo "Done."
date

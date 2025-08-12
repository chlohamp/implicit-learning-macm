#!/bin/bash
#SBATCH --job-name=macm
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=16gb
#SBATCH --account=iacc_nbc
#SBATCH --qos=pq_nbc
#SBATCH --partition=IB_40C_512G
# Outputs ----------------------------------
#SBATCH --output=/home/data/nbc/misc-projects/crooks_macm/code/log/macm/%x_%j.out
#SBATCH --error=/home/data/nbc/misc-projects/crooks_macm/code/log/macm/%x_%j.err
# ------------------------------------------

# IB_44C_512G, IB_40C_512G, IB_16C_96G, for running workflow
# investor, for testing
pwd; hostname; date

#==============Shell script==============#

# Load evironment
module add miniconda3-4.5.11-gcc-8.2.0-oqs2mbg

# Load evironment
source /home/data/nbc/misc-projects/Hampson_Habenula/env/activate_env

set -e

PROJECT_DIR="/home/data/nbc/misc-projects/crooks_macm/"
CODE_DIR=${PROJECT_DIR}/code

# Setup done, run the command
cmd="python ${CODE_DIR}/macm_workflow.py \
    --project_dir ${PROJECT_DIR} \
    --n_cores ${SLURM_CPUS_PER_TASK}"
echo Commandline: $cmd
eval $cmd

# Output results to a table
echo Finished tasks with exit code $exitcode
exit $exitcode

date

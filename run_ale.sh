#!/bin/bash
#SBATCH --job-name=SDMA-workflow-point
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --account=iacc_nbc
#SBATCH --qos=pq_nbc
#SBATCH --partition=IB_40C_512G
# Outputs ----------------------------------
#SBATCH --output=/home/kcroo010/SleepMeta/SStruct/jobs/%x_%j.out
#SBATCH --error=/home/kcroo010/SleepMeta/SStruct/jobs/%x_%j.err
# ------------------------------------------

# IB_44C_512G, IB_40C_512G, IB_16C_96G, for running workflow
# investor, for testing
pwd; hostname; date

#==============Shell script==============#


## this is for the other folder##
# Load evironment
module add miniconda3-4.5.11-gcc-8.2.0-oqs2mbg
echo "Hello World"
set -e

PROJECT_DIR="/home/kcroo010/SleepMeta/SStruct"

# Setup done, run the command
cmd="python ${PROJECT_DIR}/workflow.py \
    --project_dir ${PROJECT_DIR} \
    --n_cores ${SLURM_CPUS_PER_TASK}"
echo Commandline: $cmd
eval $cmd

# Output results to a table
echo Finished tasks with exit code $exitcode
exit $exitcode

date


# !/bin/sh
# Grid Engine options (lines prefixed with #$)
#$ -N random_forest_bert
# runtime limit of 48 hrs
#$ -l h_rt=48:00:00
# set working directory to the directory of the MLP
#$ -wd /exports/eddie/scratch/s1929142/MLP
# set the output and error stream to the output directory:
#$ -o /exports/eddie/scratch/s1929142/MLP/output/random_forest_bert_output.txt
#$ -e /exports/eddie/scratch/s1929142/MLP/output/random_forest_bert_error.txt
# request 80 GB system RAM per GPU
#$ -l h_vmem=80G
#  These options are:
#  job name: -N
#  use the current working directory: -cwd
#  runtime limit of 5 minutes: -l h_rt
#  memory limit of 1 Gbyte: -l h_vmem
# request one GPU in the GPU queue:
#$ -q gpu
#$ -pe gpu-a100 1

# Initialise the environment modules
. /etc/profile.d/modules.sh
module load cuda

# Load Python
module load anaconda
source activate mlp

# Run the program
python decision_forest_BERT.py
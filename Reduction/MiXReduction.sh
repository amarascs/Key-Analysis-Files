#!/bin/bash

#Indicate the account to charge the job to
#SBATCH --account=lorenzon0

#Indicate a name to give the job. This can be anything, it's just a way to be able to easily track different jobs
#SBATCH --job-name=MiXRed_20230127_137CsRun6

#Indicate where to send information about the job
#SBATCH --mail-user=amarascs@umich.edu

#Indicate how often you want info about the job. In this case, you will receive an email when the job has ended
#SBATCH --mail-type=BEGIN,END

#Indicate how many nodes to run the job on
#SBATCH --nodes=1

#Indicate how many tasks to run on each node
#SBATCH --ntasks-per-node=1

#Indicate how many cpus to use per task
#SBATCH --cpus-per-task=1

#Indicate how much memory to use per cpu
#SBATCH --mem-per-cpu=6GB

#Indicate how long to run the job for
#SBATCH --time=00:30:00

#Indicate which partition to run the job on. In this case, we will run on the standard partition
#SBATCH --partition=standard

#SBATCH --array=0-1000

#Get rid of any modules that might still be running
#module purge

#module load python3.9-anaconda

#install required packages 
#conda activate he_nr

#Run the desired program
INPUT_DIR=/nfs/turbo/lsa-MiXturbo/MiX-2023-data/MiXData/20230127/Cs137_C3500_G3000_616_692_634_625_1136_tSumTrigger/run6
OUTPUT_DIR=/nfs/turbo/lsa-MiXturbo/MiX-2023-data/reducedData/20230127/Cs137_C3500_G3000_616_692_634_625_1136_tSumTrigger/run6
rootFile=$(ls ${INPUT_DIR}/*.root | sed -n ${SLURM_ARRAY_TASK_ID}p)
python MiXReduction.py $rootFile $OUTPUT_DIR AnalysisParams.json



#!/bin/bash
# Slurm submission script, serial job
# CRIHAN v 1.00 - Jan 2017 
# support@criann.fr

#SBATCH --share
#SBATCH --time 48:00:00
#SBATCH --mem 100000 
#SBATCH --mail-type ALL
#SBATCH --mail-user thomas.constum@insa-rouen.fr
#SBATCH --partition gpu_k80
#SBATCH --gres gpu:4
#SBATCH --nodes 1
#SBATCH --output /home/2017018/tconst01/pao/logs/%J.out
#SBATCH --error /home/2017018/tconst01/pao/logs/%J.err

module load cuda/9.0
module load python3-DL/3.6.1

export LOCAL_WORK_DIR=/home/2017018/tconst01/ssd/pao_jpeg_bis/localisation_part
export DATASET_PATH=/save/2017018/PARTAGE/pascal_voc/
#export EXPERIMENTS_OUTPUT_DIRECTORY=$HOME/experiment
export EXPERIMENTS_OUTPUT_DIRECTORY=/dlocal/run/$SLURM_JOB_ID

cd $HOME/ssd/pao_jpeg_bis/localisation_part

#srun python3 evaluation.py --ssd_resnet -pv12 -dp $DATASET_PATH --archi "ssd_custom" $HOME"/ssd/9487_epoch-258_REPRISE.h5"
#srun python3 evaluation.py --ssd_resnet -p07 -dp $HOME"/ssd/VOCdevkit/" --archi "ssd_custom" $HOME"/ssd/9487_epoch-258_REPRISE.h5"
#srun python3 evaluation.py --ssd_resnet -p07 -dp $HOME"/ssd/VOCdevkit/" --archi "deconv" $HOME"/ssd/8786_epoch-192_REPRISE.h5"
#srun python3 evaluation.py --ssd_resnet -p07 -dp $HOME"/ssd/VOCdevkit/" --archi "up_sampling" $HOME"/ssd/8789_epoch-188_REPRISE.h5"
#srun python3 evaluation.py --ssd_resnet -p07 -dp $HOME"/ssd/VOCdevkit/" --archi "y_cb4_cbcr_cb5" $HOME"/ssd/9076_epoch-231_REPRISE.h5"
srun python3 evaluation.py --ssd_resnet -p07 -dp $HOME"/ssd/VOCdevkit/" --archi "cb5_only" $HOME"/ssd/9092_epoch-166_REPRISE.h5"



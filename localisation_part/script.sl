#!/bin/bash
# Slurm submission script, serial job
# CRIHAN v 1.00 - Jan 2017 
# support@criann.fr

#SBATCH --share
#SBATCH --time 48:00:00
#SBATCH --mem 100000 
#SBATCH --mail-type ALL
#SBATCH --mail-user thomas.constum@insa-rouen.fr
#SBATCH --partition gpu_p100
#SBATCH --gres gpu:2
#SBATCH --nodes 1
#SBATCH --output /home/2017018/tconst01/pao/logs/%J.out
#SBATCH --error /home/2017018/tconst01/pao/logs/%J.err

module load cuda/9.0
module load python3-DL/3.6.1

export LOCAL_WORK_DIR=/home/2017018/tconst01/ssd/pao_jpeg_bis/localisation_part
export DATASET_PATH=/save/2017018/PARTAGE/pascal_voc/
export EXPERIMENTS_OUTPUT_DIRECTORY=/dlocal/run/$SLURM_JOB_ID

# export DATASET_PATH=/mnt/CE0452E10454CDD9/Users/Thomas/Documents/Etudes/ASI/ASI51/PAO/SSD_criann/VOCdevkit
# export EXPERIMENTS_OUTPUT_DIRECTORY=.
# export LOCAL_WORK_DIR=.

cd $HOME/ssd/pao_jpeg_bis/localisation_part

#srun python3 training_dct_pascal_j2d.py -vd 0 --crop --p07 --reg --ssd #--weights "VGG_VOC0712Plus_SSD_300x300_iter_240000.h5"
srun python3 training_dct_pascal_j2d_resnet.py -vd 0 --crop --p07p12 --reg --resnet --archi "ssd_custom" --restart $HOME"/ssd/9139_epoch-193_REPRISE.h5"
#srun python3 training_dct_pascal_j2d_resnet.py -vd 0 --crop --p07p12 --reg --resnet --archi "y_cb4_cbcr_cb5" --restart $HOME"/ssd/9076_epoch-231_REPRISE.h5" 
#srun python3 training_dct_pascal_j2d_resnet.py -vd 0 --crop --p07p12 --reg --resnet --archi "deconv" --restart $HOME"/ssd/8786_epoch-192_REPRISE.h5" 
#srun python3 training_dct_pascal_j2d_resnet.py -vd 0 --crop --p07p12 --reg --resnet --archi "cb5_only" --restart $HOME"/ssd/9092_epoch-166_REPRISE.h5"
#srun python3 training_dct_pascal_j2d_resnet.py -vd 0 --crop --p07p12 --reg --resnet --archi "up_sampling" --restart $HOME"/ssd/8789_epoch-188_REPRISE.h5"

#!/bin/bash
# Slurm submission script, serial job
# CRIHAN v 1.00 - Jan 2017 
# support@criann.fr

#SBATCH --exclusive
#SBATCH --time 48:00:00
#SBATCH --mem 100000 
#SBATCH --mail-type ALL
#SBATCH --mail-user thomas.constum@insa-rouen.fr
#SBATCH --partition gpu_k80

# GPUs per compute node
# gpu:4 (maximum) for gpu_k80
# gpu:2 (maximum) for gpu_p100
#SBATCH --gres gpu:4

# Compute nodes number
#SBATCH --nodes 4
#SBATCH --output /home/2017018/tconst01/pao/logs/%J.out
#SBATCH --error /home/2017018/tconst01/pao/logs/%J.err

# MPI tasks per compute node
# même nombre que les gpus : (4 si k80, 2 si p100)
#SBATCH --tasks-per-node=4

# CPUs per MPI task (= OMP_NUM_THREADS)
# 7 pour k80 (4 workers), 14 pour les p100
#SBATCH --cpus-per-task=7

module load cuda/9.0
module load python3-DL/3.6.1

export PYTHONUSERBASE=$HOME/ssd/pao_jpeg_bis/classification_part
#export EXPERIMENTS_OUTPUT_DIRECTORY=$HOME/experiment
export EXPERIMENTS_OUTPUT_DIRECTORY=/dlocal/run/$SLURM_JOB_ID

export LOG_DIRECTORY=$HOME/pao/logs/
export DATASET_PATH_TRAIN=/save/2017018/PARTAGE/
export DATASET_PATH_VAL=/save/2017018/PARTAGE/
export PROJECT_PATH=$HOME/ssd/pao_jpeg_bis/classification_part

cd $HOME/ssd/pao_jpeg_bis/classification_part

# We re install the package
#srun python3 $HOME/vgg_final/vgg_jpeg/keras/training.py -c $HOME/vgg_final/vgg_jpeg/keras/config/vggA_dct True
# srun python3 $HOME/vgg_final/vgg_jpeg/keras/training.py -c $HOME/vgg_final/vgg_jpeg/keras/config/resnet True --deconv True
#srun python3 $HOME/vgg_final/vgg_jpeg/keras/training_copy.py -c $HOME/vgg_final/vgg_jpeg/keras/config/originalResnet True
#srun python3 $HOME/vgg_final/vgg_jpeg/keras/training.py -c $HOME/vgg_final/vgg_jpeg/keras/config/vggD_dct True

srun python3 training.py -c config/resnetRGB --archi "resnet_rgb" --use_pretrained_weights "True" --horovod "True"
# python3 training.py -c config/resnet --archi "deconv" --use_pretrained_weights "True" --horovod "False"
# python3 training.py -c config/resnet --archi "cb5_only" --use_pretrained_weights "True" --horovod "False"
# python3 training.py -c config/resnet --archi "up_sampling" --use_pretrained_weights "True" --horovod "False"
# python3 training.py -c config/resnet --archi "up_sampling_rfa" --use_pretrained_weights "True" --horovod "False"
# python3 training.py -c config/resnet --archi "y_cb4_cbcr_cb5" --use_pretrained_weights "True" --horovod "False"
# python3 training.py -c config/resnet --archi "late_concat_rfa_thinner" --use_pretrained_weights "True" --horovod "False"
# python3 training.py -c config/resnet --archi "late_concat_more_channels" --use_pretrained_weights "True" --horovod "False"
# python3 training.py -c config/resnetRGB --archi "resnet_rgb" --use_pretrained_weights "True" --horovod "False"
# python3 training.py -c config/vggA_dct --archi "vggA_dct" --use_pretrained_weights "False" --horovod "False"
# python3 training.py -c config/vggD_dct --archi "vggD_dct" --use_pretrained_weights "False" --horovod "False"

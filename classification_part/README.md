# INSA Rouen Project - Neural network for object detection in JPEG images

This project aims to train a classification neural network using compressed JPEG files.

## How to install

```bash

git submodule init
git submodule update

cd deep_template/template_keras/template_keras

pip install . --user

cd ../../..

pip install -r requirements.txt --user
```

## How to start the calculations: Slurm

```bash
# Use the pre-set config file (training neural network in RGB) : without distributed computation
python training.py -c config/vggA/

# Use the provided slurm script (modify to your needs)
sbatch vgg_jpeg.sl
# Replace the part config/vggA/ by the config file of your choice
# To change the network used, go in the config file and change the line config.network in the __init__
```

## Experiments carried out

### Classification part

#### Details on experimentations

1) Training from scratch of a customized ResNet50 able to handle DCT inputs. The modified layers follow the architecture "Late Concat RFA thinner" explained in the UBER article.
The results are better than VGG DCT but far from original ResNet results

2) Training from scratch of ResNet50 in RGB space using the data augmentation of the original Benjamin's code. The results are quite the same as 1) which is weird

3) Same training as 2) but this time the model loads pretained weight of ResNet50 "resnet50\_weights\_tf\_dim\_ordering\_tf\_kernels.h5". The results are way better and pretty close from the original ResNet results. 
   These better results are probably due to the fact that the loaded weights are perfectly fine-tunes and then help the SGD to better to converge.

4) Same training as 1) but this time the model loads pretained weight of ResNet50 "resnet50\_weights\_tf\_dim\_ordering\_tf\_kernels.h5" on the blocks 3 to 5. To do this, I had to modify the number of channels in several layers to have the original number of channels in the blocks 3 to 5.
   The numbers of channels in the last layer of the blocks 1 and 2 of luma were changed from 768 to 384. The number of channels in the last layer of the block of CrCb was changed from 256 to 128.
   These changes are mandatory for the number of input channels in block 3 to be 512. With this modification, the channels of the resnet layers are identical to the original resnet, the
   it is then possible to load the pretrained weight of original ResNet50.
   The results are pretty good, not as good as pretained ResNetRGB but better that both VggDct and ResnetRGB from scratch.
   
5) Since I had to decrease the number of channels for the blocks 3 to 5, I wanted to see if increasing the number of channels of the blocks 1 and 2 was making a difference. 
   So I changed the number of channels of each block to [256, 256, 768]. This increase in the number of channels does not have a significant influence on the val accuracy. This is probably due to the fact that the most part of the training is already done by the pretrained weights.
   
6) I tried to use the same parameters of SGD used for the original ResNet which are lr=0.1 and decay=0.0001, instead of the lr=0.01 and decay=0 used previously. 
   The difference is not significative but a bit better. The low difference is probably due to the same reasons as in 5)
   
7) Same as 6) but the architecture of the first layers of Y and CbCr are changed to follow the architecture "Up Sampling RFA" from Über. The results are better than 6).
   This is consistant with the Über article which shows a difference of 0.3% of accuracy on average between "Up Sampling RFA" and "Late Concat RFA Thinner".
   
#### Table of results

| Architecture                        | Top 1 training | Top 5 training | Top 1 val | Top 5 val | Loss |
|-------------------------------------|----------------|----------------|-----------|-----------|------|
| Original Resnet50         |                |                | 0.793  | 0.948    |      |
| ResNet DCT from Uber (Late Concat RFA Thinner)             |                |                | 0.754  | 0.930    |      |
| VGG DCT from Benjamin               |                |                | 0.420     | 0.669     |      |
| 1) Resnet50 DCT from scratch           | 0.734          | 0.911        | 0.525     | 0.768     | 2.13 |
| 2) Resnet50 RGB from scratch           | 0.733          | 0.922          | 0.519     | 0.761     | 2.20 |
| 3) Resnet50 RGB with pretained weights | 0.875          | 0.981         | 0.675     | 0.885     | 1.44 |
| 4) Resnet50 DCT with pretained weights |     0.908           |       0.986         |       0.612    |     0.834      |    1.83  |
| 5) Resnet50 DCT with pretained weights + more channels |     0.884           |       0.980         |       0.618    |     0.838      |    1.75  |
| 6) 5) with lr=0.1 and decay=0.0001 |         0.928       |     0.990   |     0.621  |     0.843     | 1.81   |
| 7) Up Sampling RFA |         0.946       |     0.994   |     0.678  |     0.878     | 2.104   |

### Detection part

#### Details on experimentations

#### Table of results

| Architecture                        | mAP | Training loss| Val loss |
|-------------------------------------|----------------|----------------|-----------|
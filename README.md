# Neural networks for object detection in JPEG images

This project aims to train a neural network to detect objects in JPEG images. Several architectures were trained for classification then the weights were used to fine-tune a SSD based network for object detection.
This project was carried out as an optional project at INSA Rouen Normandy and was supervised by Benjamin Deguerre.
The project continue the research of [Benjamin Deguerre](https://github.com/D3lt4lph4) on object detections in JPEG images using a VGG backbone.
This project experiments namely the use of Resnet as a backbone for detection.

### Reference articles : 
* Fast object detection in compressed JPEG Images (INSA Rouen Normandy) : https://arxiv.org/abs/1904.08408 (Benjamin Deguerre, Clément Chatelain, Gilles Gasso)
* Faster Neural Networks Straight from JPEG (Über) : https://papers.nips.cc/paper/7649-faster-neural-networks-straight-from-jpeg (Lionel Gueguen, Alex Sergeev, Ben Kadlec, Rosanne Liu, Jason Yosinski)

**This project was originally hosted on Gitlab INSA Rouen at [this adress](https://gitlab.insa-rouen.fr/tconstum/pao_jpeg_bis)**
**The results of this project have then been improved by Benjamin Deguerre and led to a new publication published on June 10th ["Object Detection in the DCT Domain: is Luminance the Solution?"](https://arxiv.org/abs/2006.05732). The corresponding code was also published [here](https://github.com/D3lt4lph4/jpeg_deep)**

## Contents

1. [How to insall](#how-to-install)
2. [How to use](#how-to-use)
3. [Experiments carried out](#experiments-carried-out)

## How to install

### Classification part

At the root of the repository :

```bash
git submodule init
git submodule update 
cd classification_part
pipenv shell
pipenv install --skip-lock
cd jpeg2dct
python3 setup.py install
```

### Localization part

At the root of the repository :

```bash
git submodule init
git submodule update    
cd localisation_part
pipenv shell
pipenv install --skip-lock
cd jpegdecoder 
git submodule init
git submodule update 
python3 setup.py install
cd ../jpeg2dct
python3 setup.py install
```

## How to use

### Classification part

Before using the program, several environment variables have to be set :

* EXPERIMENTS_OUTPUT_DIRECTORY : the path were the trained weights will be stored
* LOG_DIRECTORY : the path were the log files will be stored (one for stderr and one for stdout)
* DATASET_PATH_TRAIN : the path of the train set of imagenet 
* DATASET_PATH_VAL : the path of the evaluation set of imagenet 
* PROJECT_PATH : the absolute path of this current project (useful for Slurm)

For example : 

```bash
export EXPERIMENTS_OUTPUT_DIRECTORY=/dlocal/run/$SLURM_JOB_ID
export LOG_DIRECTORY=$HOME/logs
export DATASET_PATH_TRAIN=/save/2017018/PARTAGE/
export DATASET_PATH_VAL=/save/2017018/PARTAGE/
export PROJECT_PATH=$HOME/vgg_jpeg
```


Indicate the network architecture in the "--archi" argument :
* cb5_only : CbCr and Y are only going through the conv block 5 of Resnet50
* deconv : deconvolution architecture of Über article
* up_sampling : up sampling architecture of Über article
* up_sampling_rfa : up sampling RFA architecture of Über article
* y_cb4_cbcr_cb5 : Y go through the conv block 4 of Resnet50 and CbCr go through the conv block 5
* late_concat_rfa_thinner : late concat RFA thinner architecture of Über article
* late_concat_more_channels : late concat RFA thinner architecture of Über article with more channels
* resnet_rgb : regular Resnet50 RGB
* vggA_dct : VGG A DCT architecture from Benjamin Deguerre's research paper
* vggD_dct : VGG D DCT architecture from Benjamin Deguerre's research paper

Indicate the path to the config file to use. The configuration file should be named 'config_file.py'
For example : 
```bash
python3 training.py -c config/resnet --archi "deconv" --use_pretrained_weights "True" --horovod "False"
```

Horovod is for multi-GPU only. An example of execution with Slurm and Horovod can be found in the script slurm "vgg_jpeg.sl".
The argument --use_pretrained_weights indicates if you want to use the pretrained weights from Keras. (resnet50_weights_tf_dim_ordering_tf_kernels.h5)

By default, the SGD parameters are : 
```python
self.optimizer_parameters = {"lr": 0.01, "momentum": 0.9, "decay": 0, "nesterov": True}
```

### Localisation part

#### Training

Before using the program, several environment variables have to be set :

* EXPERIMENTS_OUTPUT_DIRECTORY : the path were the trained weights will be stored. If it is not set, the default value will be "./n" with n being the number of the GPU device indicated in --vd. (explanations below)
* DATASET_PATH: the path of the Pascal VOC dataset
* LOCAL_WORK_DIR : the absolute path of this current project (useful for Slurm)

For example : 

```bash
export LOCAL_WORK_DIR=/home/2017018/user/ssd/pao_jpeg
export DATASET_PATH=/save/2017018/PARTAGE/pascal_voc/
export EXPERIMENTS_OUTPUT_DIRECTORY=/dlocal/run/$SLURM_JOB_ID
```

To train each of the following architectures, the program has to be called using the command : 
```bash
python3 training_dct_pascal_j2d_resnet.py -vd 0 --crop --p07p12 --reg --resnet --archi "ssd_custom" --restart "../2564_epoch-133.h5" 
```

Choose between --p07 for the Pascal VOC 2007 and --p07p12 for the evaluation set of both Pascal VOC 2007 and 2012 combined.
Add the "--restart" argument to restart a training from pretrained weights. Since there is no multi-GPU support for this part the training can be pretty long (more than 48 hours on a Nvidia P100).
It is then convenient to separate the training in several sessions.
Indicate the network architecture in the "--archi" argument :
* "ssd_custom" : the extra-feature layers of SSD are removed to match dimension with full Late-concat-RFA architecture of Über (Experiment one)
* "y\_cb4\_cbcr_cb5" Y go through the conv block 4 of Resnet50 and CbCr go through conv block 5 (Experiment 2)
* "cb5_only" CbCr and Y are only going through the conv block 5 of Resnet50 (Experiment 3)
* "up_sampling" up sampling RFA architecture of Über article (Experiment 4)
* "deconv" deconvolution architecture of Über article (Experiment 5)


The results and weights are stored in the folder corresponding to the GPU device number indicated for the argument "-vd". For the command above it would be "./0". 

#### Evaluation

To launch the evaluation, the same environment variables as above have to be set.
To launch the evaluation on the Pascal VOC 2007 test set :
```bash
python3 evaluation.py -p07 --archi "cb5_only" "../experiment_k_epoch-N.h5"
```

The datasets and architectures available are the same as for training
The results of evaluation will be stored in ```EXPERIMENTS_OUTPUT_DIRECTORY ```

## Experiments carried out

### Classification part

#### Details on experimentations

**Implementation of the following architectures can be found [here](classification_part/vgg_jpeg_keras/networks/resnet_dct.py)**

1) Training from scratch of a customized ResNet50 able to handle DCT inputs. The modified layers follow the architecture "Late Concat RFA thinner" explained in the UBER article.
The results are better than VGG DCT but far from original ResNet results.

2) Training from scratch of ResNet50 in RGB space using the data augmentations of the original Benjamin's code. The results are quite the same as 1) when they should be better.

3) Same training as 2) but this time the model loads pretained weights of ResNet50 "resnet50\_weights\_tf\_dim\_ordering\_tf\_kernels.h5". The results are way better and pretty close from the original ResNet results. 
   These better results are probably due to the fact that the loaded weights are perfectly fine-tuned and then help the SGD to better converge.

4) Same training as 1) but this time the model loads pretained weights of ResNet50 "resnet50\_weights\_tf\_dim\_ordering\_tf\_kernels.h5" on the blocks 3 to 5. To do this, I had to modify the number
   of channels in several layers to get the original number of channels in the blocks 3 to 5.
   The numbers of channels in the last layer of the blocks 1 and 2 of Y were changed from 768 to 384. The number of channels in the last layer of the block of CrCb was changed from 256 to 128.
   These changes are mandatory for the number of input channels in block 3 to be 512 (384+128). With this modification, the number of channels of the resnet layers are identical to the original resnet.
   It is then possible to load the pretrained weights of original ResNet50.
   The results are pretty good, not as good as pretained ResNetRGB but better than VggDct and ResnetRGB trained from scratch.
   
5) Since I had to decrease the number of channels for the blocks 3 to 5, I wanted to see if increasing the number of channels of the blocks 1 and 2 was making a difference. 
   So I changed the number of channels of each block to [256, 256, 768]. This increase in the number of channels does not have a significant influence on the val accuracy. 
   This is probably due to the fact that the most part of the training is already done by the pretrained weights.
   
6) I tried to use the same parameters of SGD used for the original ResNet which are lr=0.1 and decay=0.0001, instead of the lr=0.01 and decay=0 used previously. 
   The difference is not significative but a bit better. The low difference is probably due to the same reasons as in 5)
   
7) Same as 6) but the architecture of the first layers of Y and CbCr are changed to follow the architecture "Up Sampling RFA" from Über. The results are better than 6).
   This is consistant with the Über article which shows a difference of 0.3% of accuracy on average between "Up Sampling RFA" and "Late Concat RFA Thinner".
   
8) The architecture is the same as 5) until CbCr and Y are concatenated. Then, the information only go through CB5 of original Resnet.
   The results are not that bad compared to other expriments and regarding the shallowness of the architecture. This architecture is useful for detection part because 
   it avoids dimensions problems and then allows to keep the extra-feature layers of SSD unchanged. (see explanations in detection part)
   
9) Another architecture experimented for object detection. To avoids dimensions problems with SSD, CBCR is concatenated to Y between CB4 and CB5.
   Thus, CbCr only go through CB5 and then colour is not taken into account by CB4.
   
10) Experimentation of the deconvolution-RFA architecture of the Über article. The results are the best of all the architectures tested which is consistent with the results of Über.

#### Table of results

| Architecture                        | Top 1 training | Top 5 training | Top 1 val | Top 5 val | Loss |
|-------------------------------------|----------------|----------------|-----------|-----------|------|
| Original Resnet50         |                |                | 0.793  | 0.948    |      |
| ResNet DCT from Uber (Late Concat RFA Thinner)             |                |                | 0.754  | 0.930    |      |
| VGG DCT from Benjamin Deguerre              |                |                | 0.420     | 0.669     |      |
| 1) Resnet50 DCT from scratch           | 0.734          | 0.911        | 0.525     | 0.768     | 2.13 |
| 2) Resnet50 RGB from scratch           | 0.733          | 0.922          | 0.519     | 0.761     | 2.20 |
| 3) Resnet50 RGB with pretained weights | 0.875          | 0.981         | 0.675     | 0.885     | 1.44 |
| 4) Resnet50 DCT with pretained weights |     0.908           |       0.986         |       0.612    |     0.834      |    1.83  |
| 5) Resnet50 DCT with pretained weights + more channels |     0.884           |       0.980         |       0.618    |     0.838      |    1.75  |
| 6) 5) with lr=0.1 and decay=0.0001 |         0.928       |     0.990   |     0.621  |     0.843     | 1.81   |
| 7) Up Sampling RFA |         0.946       |     0.994   |     0.678  |     0.878     | 2.104   |
| 8) CB5 only |         0 .885      |     0.978   |     0.614  |     0.839     |  1.7323  |
| 9) y in CB4, cbcr in CB5 |         0 .856      |     0.968   |     0.608  |     0.832     |  1.7425  |
| 10) Deconv |         0 .916      |     0.988   |     0.684  |     0.887     |  1.3978  |


### Detection part

#### Details on experimentations

**Implementation of the following architectures can be found [here](localisation_part/models/keras_ssd300_dct_j2d_resnet.py)**

1) The convolutionnal blocks of VGG are replaced with the convolutionnal blocks from 3 to 5 of Resnet. Somes changes are needed 
   for the extra feature layers. Indeed, the Late-Concat-RFA-Thinner architecture lead to a smaller output layer size than VGG, which means that some layers has to be removed.
   The last layers of the original SSD lead to a layer size of 1\*1, which means that the output of resnet layers cannot be smaller that the VGG ones if the extra-feature layers of SSD are kept unchanged.
   The extra-feature layers from 6\_2 to 8\_2 are removed to avoid dimension errors. 
   The relation between old and new extra feature layers is as follows :
   * conv4\_3 stays conv4_3 (VGG notation)
   * fc7 becomes conv3_3 (resnet)
   * conv6\_2 becomes conv4_6 (resnet)
   * conv7_2 becomes fc7 (Ssd notation)
   * conv8\_2 becomes conv6_2 (Ssd notation)

   The mAP obtained with this architecture is the best obtained compared to other methods.
   
2) The non-VGG SSD layers are kept unchanged. The architecture experimented takes as baseline the late concat RFA thinner
   architecture but the CB3 is removed and the stride of the last layer of Y becomes (1,1). Moreover, the CbCr is concatenated with Y between CB4 and
   CB5 instead of between CB3 and CB4 which means that only Y goes through CB4 and not CbCR.
   
3) This architecture follows the "Late Concat RFA Thinner" architecture but the CB3 and CB4 are removed. Once concatenated,
   Y and CbCr only go through CB5 before going through the extra feature layers. The extra feature layers are kept unchanged.
   
4) This architecture follows the "Up Sampling RFA" architecture from Über for the classification backbone. The extra feature layers are kept unchanged.

5) This architecture follows the "Deconvolution RFA" architecture from Über for the classification backbone. The extra feature layers are kept unchanged.

It appears from the results that the modification of extra-feature layers is more beneficial for the mAP than modifying the classication
backbone to match dimensions. Indeed, even if the deconvolution network shows the best accuracy in classification, the best mAP
is obtained with the ssd_custom architecture where extra features layers are replaced with Resnet layers.
Some classes are particularly better recognized with ssd_custom. This is the case with the classes car, cat sheep and tv monitor.
Yet, some classes are better recognized with other architecture than ssd_custom. This is especially the case for the classes bird, cat and train
which are better recognized by deconv or up sampling architectures.
The results are quite the same between PV val 2012 and PV test 2007 except for the SSD custom architecture which gains almost 4% of mAP.

#### Summary of results

| Architecture                        | mAP on Pascal VOC val 2012 (%)| mAP on Pascal VOC test 2007 (%)|Training loss| Val loss |
|-------------------------------------|----------------|----------------|-----------|-|
|SSD DCT with VGG||47.8|||
| 1) SSD with CB3 to CB5 Resnet (custom SSD)|59.2|63|3.4521|3.9092|
| 2) y in CB4, CbCr in CB5|48|48.5|4.5473|4.8983|
| 3) CB3 and CB4 removed (CB5 only)|47.4|47.7|4.8709|5.1001|
| 4) Up sampling|51.2|51.9|4.5622|4.755|
| 5) Deconvolution|51.5|52.7|4.457|4.673|

#### Details of results

##### Results on Pascal VOC evaluation set 2012

||ssd custom|up_samplig|cbcr cb5|deconv|cb5 only|
|--------|----------------|----------------|-----------|-|-|
aeroplane|0,774|0,712|0,687|0,733|0,705
bicycle|0,696|0,597|0,574|0,592|0,575
bird|0,528|0,483|0,434|0,514|0,445
boat|0,425|0,322|0,304|0,329|0,304
bottle|0,245|0,174|0,153|0,183|0,144
bus|0,754|0,714|0,695|0,688|0,688
car|0,576|**0,37**|**0,35**|**0,372**|**0,344**
cat|0,812|**0,838**|0,775|**0,845**|0,779
chair|0,329|0,226|0,182|0,229|0,173
cow|0,54|0,486|0,443|0,49|0,423
diningtable|0,499|0,468|0,464|0,456|0,446
dog|0,744|0,739|0,652|0,733|0,674
horse|0,714|0,666|0,639|0,672|0,627
motorbike|0,727|0,641|0,623|0,653|0,633
person|0,674|0,521|0,492|0,536|0,479
pottedplant|0,292|0,182|0,158|0,189|0,142
sheep|0,594|**0,376**|**0,336**|**0,366**|**0,332**
sofa|0,554|0,522|0,502|0,53|0,492
train|0,759|0,789|0,74|0,755|0,719
tvmonitor|0,602|0,422|**0,392**|0,437|**0,364**
mAP|0,592|0,512|0,48|0,515|0,474|

##### Results on Pascal VOC test set 2007

||ssd custom|up_samplig|cbcr cb5|deconv|cb5 only|
|--------|----------------|----------------|-----------|-|-|
aeroplane|0,705|0,559|0,531|0,55|0,548
bicycle|0,737|0,556|0,56|0,563|0,547
bird|0,548|0,478|0,407|0,485|0,415
boat|0,526|0,418|0,373|0,398|0,351
bottle|0,209|0,105|0,09|0,11|0,064
bus|0,766|0,691|0,656|0,708|0,646
car|0,744|**0,503**|**0,49**|**0,52**|**0,485**
cat|0,784|**0,802**|0,744|**0,811**|0,763
chair|0,348|0,178|**0,131**|0,19|**0,123**
cow|0,584|0,412|**0,376**|0,435|**0,349**
diningtable|0,648|0,648|0,615|0,64|0,596
dog|0,761|0,752|0,678|0,731|0,674   
horse|0,81|0,744|0,716|0,735|0,729
motorbike|0,778|0,632|0,601|0,636|0,589
person|0,661|0,484|**0,457**|0,498|**0,438**
pottedplant|0,295|**0,176**|**0,151**|0,202|**0,142**
sheep|0,601|0,378|0,38|0,416|0,37
sofa|0,655|0,628|0,573|0,615|0,587
train|0,807|0,788|0,733|**0,81**|0,745
tvmonitor|0,622|0,448|0,428|0,484|**0,375**
mAP|0,63|0,519|0,485|0,527|0,477


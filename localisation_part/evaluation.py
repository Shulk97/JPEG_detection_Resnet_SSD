from keras import backend as K
from keras.models import load_model
from keras.optimizers import Adam, SGD
from scipy.misc import imread
import numpy as np
import matplotlib
import csv

from models.keras_ssd300 import ssd_300
from models.keras_ssd300_dct_j2d_resnet import ssd_resnet_EF_layers_identical, ssd_resnet_EF_layers_custom
from models.keras_ssd300_dct_no_regularizer import ssd_300DCT_no_reg as ssd_miisst_dct
from models.keras_ssd300_dct_j2d_no_regularizer import ssd_300DCT_no_reg as ssd_300DCT
from keras_loss_function.keras_ssd_loss import SSDLoss
from keras_layers.keras_layer_AnchorBoxes import AnchorBoxes
from keras_layers.keras_layer_DecodeDetections import DecodeDetections
from keras_layers.keras_layer_DecodeDetectionsFast import DecodeDetectionsFast
from keras_layers.keras_layer_L2Normalization import L2Normalization
from data_generator.object_detection_2d_data_generator_dct_j2d import DataGeneratorDCT,DataGeneratorDeconvDCT

from data_generator.object_detection_2d_data_generator import DataGenerator

from eval_utils.average_precision_evaluator import Evaluator

from argparse import ArgumentParser
import os

parser = ArgumentParser()
parser.add_argument("weights", type=str)
parser.add_argument("-r", "--ssd_resnet", action='store_true', default=False)
parser.add_argument("-s", "--ssd", action='store_true', default=False)
parser.add_argument("-so", "--ssd_other", action='store_true', default=False)
parser.add_argument("-sd", "--ssd_dct", action='store_true', default=False)
parser.add_argument("-sm", "--ssd_miisst", action='store_true', default=False)
parser.add_argument("-smd", "--ssd_miisst_dct", action='store_true', default=False)
parser.add_argument("-p12", "--pascal_2012", action='store_true', default=False)
parser.add_argument("-p10", "--pascal_2010", action='store_true', default=False)
parser.add_argument("-pv12", "--pascal_val_2012", action='store_true', default=False)
parser.add_argument("-p07", "--pascal_2007", action='store_true', default=False)
parser.add_argument("-mv", "--miisst_val", action='store_true', default=False)
parser.add_argument("-mt", "--miisst_train", action='store_true', default=False)
parser.add_argument("-dp", "--dataset_path")
parser.add_argument("--archi", help="""The network architecture to use, value can be :\n
* cb5_only : CbCr and Y only go through the conv block 5 of Resnet50\n
* deconv : deconvolution architecture of Über article\n
* up_sampling : up sampling architecture of Über article\n
* y_cb4_cbcr_cb5 :Y go through the conv block 4 of Resnet50 and CbCr go through conv block 5\n
* ssd_custom : the extra-feature layers of SSD are removed to match dimension with full Late-concat-RFA architecture of Über
""")
args = parser.parse_args()

# Set a few configuration parameters.
img_height = 300
img_width = 300
if args.ssd_miisst or args.ssd_miisst_dct:
    n_classes = 3
else:
    n_classes = 20
model_mode = 'inference'

# 1: Build the Keras model

K.clear_session() # Clear previous models from memory.

ssd_params = {"image_size":(img_height, img_width, 3),
                "n_classes":n_classes,
                "mode":model_mode,
                "l2_regularization":0.0005,
                "scales":[0.1, 0.2, 0.37, 0.54, 0.71, 0.88, 1.05], # The scales for MS COCO [0.07, 0.15, 0.33, 0.51, 0.69, 0.87, 1.05]
                "aspect_ratios_per_layer":[[1.0, 2.0, 0.5],
                                         [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                                         [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                                         [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                                         [1.0, 2.0, 0.5],
                                         [1.0, 2.0, 0.5]],
                "two_boxes_for_ar1":True,
                "steps":[8, 16, 32, 64, 100, 300],
                "offsets":[0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                "clip_boxes":False,
                "variances":[0.1, 0.1, 0.2, 0.2],
                "normalize_coords":True,
                "subtract_mean":[123, 117, 104],
                "swap_channels":[2, 1, 0],
                "confidence_thresh":0.01,
                "iou_threshold":0.45,
                "top_k":200,
                "nms_max_output_size":400,
                "archi":args.archi}


if args.archi == "ssd_custom":
    model = ssd_resnet_EF_layers_custom(**ssd_params)
else:
    model = ssd_resnet_EF_layers_identical(**ssd_params)
# 2: Load the trained weights into the model.

weights_path = args.weights 

model.load_weights(weights_path)

# 3: Compile the model so that Keras won't complain the next time you load it.

adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
sgd = SGD(lr=0.001, momentum=0.9, decay=0.0, nesterov=False)

ssd_loss = SSDLoss(neg_pos_ratio=3, alpha=1.0)

model.compile(optimizer=adam, loss=ssd_loss.compute_loss)

if args.archi == "deconv":
    print("Using generator for deconvolution network (Y, Cb and Cr separated)")
    dataset = DataGeneratorDeconvDCT()
elif args.ssd_dct or args.ssd_miisst_dct or args.ssd_resnet:
    print("Using generator for standard DCT architectures (SSD based on VGG or Resnet)")
    dataset = DataGeneratorDCT()
else:
    print("Using standard RGB generator")
    dataset = DataGenerator()

evaluate_pred = True
# TODO: Set the paths to the dataset here.
if args.ssd_miisst or args.ssd_miisst_dct:
    print("Setting the dataset to miisst test set.")
    Pascal_VOC_dataset_images_dir = os.path.join(args.dataset_path,'/MIISST_camera_snapshots/images')
    Pascal_VOC_dataset_annotations_dir = os.path.join(args.dataset_path,'/MIISST_camera_snapshots/xmls')
    Pascal_VOC_dataset_image_set_filename = os.path.join(args.dataset_path,'/MIISST_camera_snapshots/sets/test.txt')
    if args.miisst_val:
        print("Setting the dataset to miisst val set.")
        Pascal_VOC_dataset_image_set_filename = os.path.join(args.dataset_path,'/MIISST_camera_snapshots/sets/val.txt')
    if args.miisst_train:
        print("Seeting the dataset to miisst train set.")
        Pascal_VOC_dataset_image_set_filename = os.path.join(args.dataset_path,'/MIISST_camera_snapshots/sets/train.txt')
else:
    if args.pascal_2012:
        print("Setting the dataset to PASCAL VOC 2012 test set.")
        Pascal_VOC_dataset_images_dir = os.path.join(args.dataset_path,'VOC2012_test/JPEGImages/')
        Pascal_VOC_dataset_annotations_dir = None 
        Pascal_VOC_dataset_image_set_filename = os.path.join(args.dataset_path,'VOC2012_test/ImageSets/Main/test.txt')
        evaluate_pred = False
    elif args.pascal_2010:
        Pascal_VOC_dataset_images_dir = os.path.join(args.dataset_path,'VOC2010_test/JPEGImages/')
        Pascal_VOC_dataset_annotations_dir = None 
        Pascal_VOC_dataset_image_set_filename = os.path.join(args.dataset_path,'VOC2010_test/ImageSets/Main/test.txt')
        evaluate_pred = False
    elif args.pascal_val_2012:
        Pascal_VOC_dataset_images_dir = os.path.join(args.dataset_path,'VOC2012/JPEGImages/')
        Pascal_VOC_dataset_annotations_dir = os.path.join(args.dataset_path,'VOC2012/Annotations/')
        Pascal_VOC_dataset_image_set_filename = os.path.join(args.dataset_path,'VOC2012/ImageSets/Main/val.txt')
    else:
        print("Setting the dataset to PASCAL VOC 2007 test set")
        Pascal_VOC_dataset_images_dir = os.path.join(args.dataset_path,'VOC2007_test/JPEGImages/')
        Pascal_VOC_dataset_annotations_dir = os.path.join(args.dataset_path,'VOC2007_test/Annotations/')
        Pascal_VOC_dataset_image_set_filename = os.path.join(args.dataset_path,'VOC2007_test/ImageSets/Main/test.txt')


# The XML parser needs to now what object class names to look for and in which order to map them to integers.
if args.ssd_miisst or args.ssd_miisst_dct:
    classes = ['background',
            'car', 'truck', 'motorcycle']
else:
    classes = ['background',
            'aeroplane', 'bicycle', 'bird', 'boat',
            'bottle', 'bus', 'car', 'cat',
            'chair', 'cow', 'diningtable', 'dog',
            'horse', 'motorbike', 'person', 'pottedplant',
            'sheep', 'sofa', 'train', 'tvmonitor']

# 

dataset.parse_xml(images_dirs=[Pascal_VOC_dataset_images_dir],
                  image_set_filenames=[Pascal_VOC_dataset_image_set_filename],
                  annotations_dirs=[Pascal_VOC_dataset_annotations_dir],
                  classes=classes,
                  include_classes='all',
                  exclude_truncated=False,
                  exclude_difficult=True,
                  ret=False)
print("Initialize evaluator : ")
evaluator = Evaluator(model=model,
                      n_classes=n_classes,
                      data_generator=dataset,
                      model_mode=model_mode)
print("Evaluator initialized")
if evaluate_pred:
    results = evaluator(img_height=img_height,
                        img_width=img_width,
                        batch_size=8,
                        data_generator_mode='resize',
                        round_confidences=False,
                        matching_iou_threshold=0.5,
                        border_pixels='include',
                        sorting_algorithm='quicksort',
                        average_precision_mode='integrate',
                        num_recall_points=11,
                        ignore_neutral_boxes=True,
                        return_precisions=True,
                        return_recalls=True,
                        return_average_precisions=True,
                        verbose=True)

    mean_average_precision, average_precisions, precisions, recalls = results

    for i in range(1, len(average_precisions)):
        print("{:<14}{:<6}{}".format(classes[i], 'AP', round(average_precisions[i], 3)))
    print()
    print("{:<14}{:<6}{}".format('','mAP', round(mean_average_precision, 3)))

    if "EXPERIMENTS_OUTPUT_DIRECTORY" in os.environ:
        file_path = os.path.join(os.environ["EXPERIMENTS_OUTPUT_DIRECTORY"], "save/save_results.csv")
    else:
        file_path = "save_results.csv"
        
    if not os.path.exists(file_path):
        os.mkdir(os.path.join(os.environ["EXPERIMENTS_OUTPUT_DIRECTORY"], "save"))

    with open(file_path, 'w', newline='') as csvfile:
        fieldnames = ['class', 'AP']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for i in range(1, len(average_precisions)):
            writer.writerow({'class': classes[i], 'AP': round(average_precisions[i], 3)})
        writer.writerow({'class': "Moyenne", 'AP': round(mean_average_precision, 3)})

    evaluator.write_predictions_to_txt(classes=classes)
else:
    evaluator.predict_on_dataset(img_height=img_height,
                        img_width=img_width,
                        batch_size=1,
                        data_generator_mode='resize',
                        round_confidences=False,
                        verbose=True)
    evaluator.write_predictions_to_txt(classes=classes)

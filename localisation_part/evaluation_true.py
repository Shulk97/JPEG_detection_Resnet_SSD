from scipy.misc import imread
import numpy as np

from os import listdir
from os.path import isfile, join

from data_generator.object_detection_2d_data_generator_pred import DataGenerator

from eval_utils.average_precision_evaluator_pascal import Evaluator

from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("prediction_directory", type=str)
args = parser.parse_args()

# Set a few configuration parameters.
img_height = 300
img_width = 300

n_classes = 20

predictions = {}

for file in listdir(args.prediction_directory):
    if isfile(join(args.prediction_directory, file)):
        with open(join(args.prediction_directory, file)) as prediction_file:
            lines = prediction_file.readline()
            for line in lines:
                split_line = line.split()
                box = [float(line[1]), float(line[2]), float(line[3]), float(line[4]), float(line[5])]
                if line[0] not in predictions:
                    predictions[line[0]] = [box]
                else:
                    predictions[line[0]].append(box)

dataset = DataGenerator()

Pascal_VOC_dataset_images_dir = '/save/2017018/bdegue01/datasets/VOC2012/JPEGImages/'
Pascal_VOC_dataset_annotations_dir = '/save/2017018/bdegue01/datasets/VOC2012/Annotations/'
Pascal_VOC_dataset_image_set_filename = '/save/2017018/bdegue01/datasets/VOC2012/ImageSets/Main/val.txt'

# The XML parser needs to now what object class names to look for and in which order to map them to integers.
classes = ['background',
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat',
           'chair', 'cow', 'diningtable', 'dog',
           'horse', 'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor']

dataset.parse_xml(images_dirs=[Pascal_VOC_dataset_images_dir],
                  image_set_filenames=[Pascal_VOC_dataset_image_set_filename],
                  annotations_dirs=[Pascal_VOC_dataset_annotations_dir],
                  classes=classes,
                  include_classes='all',
                  exclude_truncated=False,
                  exclude_difficult=True,
                  ret=False)

evaluator = Evaluator(n_classes=n_classes,
                      data_generator=dataset,
                      predictions=predictions)

results = evaluator(img_height=img_height,
                    img_width=img_width,
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


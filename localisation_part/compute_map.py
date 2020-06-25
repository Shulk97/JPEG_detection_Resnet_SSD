import argparse
import os

import numpy as np

from tqdm import tqdm

from bs4 import BeautifulSoup

from eval_utils.utils import compute_average_precisions

parser = argparse.ArgumentParser()
parser.add_argument("--inputFolder", default="output/")

args = parser.parse_args()

def parse_xml(images_dirs,
              image_set_filenames,
              annotations_dirs=[],
              classes=None,
              exclude_difficult=False):


    if classes == None:
        classes = ['background',
                    'aeroplane', 'bicycle', 'bird', 'boat',
                    'bottle', 'bus', 'car', 'cat',
                    'chair', 'cow', 'diningtable', 'dog',
                    'horse', 'motorbike', 'person', 'pottedplant',
                    'sheep', 'sofa', 'train', 'tvmonitor']


    # Erase data that might have been parsed before.
    groundtruth = {}
    for images_dir, image_set_filename, annotations_dir in zip(images_dirs, image_set_filenames, annotations_dirs):
        print(image_set_filename)
        # Read the image set file that so that we know all the IDs of all the images to be included in the dataset.
        with open(image_set_filename) as f:
            image_ids = [line.strip() for line in f]

        it = tqdm(image_ids, desc="Processing image set '{}'".format(os.path.basename(image_set_filename)))

        # Loop over all images in this dataset.
        for image_id in it:
            # Parse the XML file for this image.
            with open(os.path.join(annotations_dir, image_id + '.xml')) as f:
                soup = BeautifulSoup(f, 'xml')

            boxes = [] # We'll store all boxes for this image here.
            objects = soup.find_all('object') # Get a list of all objects in this image.

            # Parse the data for each object.
            for obj in objects:
                class_name = obj.find('name', recursive=False).text
                class_id = classes.index(class_name)
                difficult = int(obj.find('difficult', recursive=False).text)

                # Get the bounding box coordinates.
                bndbox = obj.find('bndbox', recursive=False)
                xmin = int(bndbox.xmin.text)
                ymin = int(bndbox.ymin.text)
                xmax = int(bndbox.xmax.text)
                ymax = int(bndbox.ymax.text)

                box = [class_id, xmin, ymin, xmax, ymax, difficult]

                boxes.append(box)

            groundtruth[image_id] = boxes

    return groundtruth

images_dir = ['/save/2017018/bdegue01/datasets/VOC2007_test/JPEGImages/']
annotations_dir = ['/save/2017018/bdegue01/datasets/VOC2007_test/Annotations/']
image_set_filename = ['/save/2017018/bdegue01/datasets/VOC2007_test/ImageSets/Main/test.txt']

classes = ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

gt = parse_xml(images_dir, image_set_filename, annotations_dir)

predictions = {}
# Processing the output files to get the predictions.
for file in os.listdir(args.inputFolder):
    if os.path.isfile(os.path.join(args.inputFolder, file)):
        class_id = classes.index(file.split('_')[3][:-4])
        with open(os.path.join(args.inputFolder, file)) as open_file:
            lines = open_file.readlines()
        
        for line in lines:
            line = line.split()
            if line[0] not in predictions:
                predictions[line[0]] = []
            
            prediction = [float(value) for value in line[1:]]
            prediction.insert(0, class_id)
            predictions[line[0]].append(prediction)

prediction_boxes = []
groundtruth_boxes = []


for key in predictions:
    prediction_boxes.append(np.array(predictions[key]))
    groundtruth_boxes.append(np.array(gt[key]))

aps = compute_average_precisions(prediction_boxes, groundtruth_boxes, 20, mode="sample", ignore_under_area=None)
print(aps)
print(np.mean(aps))
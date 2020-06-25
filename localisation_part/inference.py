from keras import backend as K
from keras.models import load_model
from keras.optimizers import Adam, SGD
from imageio import imread
import numpy as np
import os
import matplotlib
matplotlib.use("qt5agg")
from matplotlib import pyplot as plt
from PIL import Image
import argparse
from models.keras_ssd300 import ssd_300
from models.keras_ssd300_dct_miisst import ssd_300DCT
from keras_loss_function.keras_ssd_loss import SSDLoss
from keras_layers.keras_layer_AnchorBoxes import AnchorBoxes
from keras_layers.keras_layer_DecodeDetections import DecodeDetections
from keras_layers.keras_layer_DecodeDetectionsFast import DecodeDetectionsFast
from keras_layers.keras_layer_L2Normalization import L2Normalization
import jpegdecoder

from ssd_encoder_decoder.ssd_output_decoder import decode_detections, decode_detections_fast

from data_generator.object_detection_2d_data_generator import DataGenerator
from data_generator.object_detection_2d_photometric_ops import ConvertTo3Channels
from data_generator.object_detection_2d_geometric_ops import Resize
from data_generator.object_detection_2d_misc_utils import apply_inverse_transforms
from data_generator.object_detection_2d_geometric_ops import Resize
from data_generator.object_detection_2d_photometric_ops import ConvertTo3Channels

parser = argparse.ArgumentParser()
parser.add_argument("-dct", "--dct", action='store_true')

args = parser.parse_args()
# Set the image size.
img_height = 300
img_width = 300

# 1: Build the Keras model

K.clear_session() # Clear previous models from memory.

if args.dct:
    model = ssd_300DCT(image_size=(img_height, img_width, 3),
                    n_classes=20,
                    mode='inference',
                    l2_regularization=0.0005,
                    scales=[0.1, 0.2, 0.37, 0.54, 0.71, 0.88, 1.05], # The scales for MS COCO are [0.07, 0.15, 0.33, 0.51, 0.69, 0.87, 1.05]
                    aspect_ratios_per_layer=[[1.0, 2.0, 0.5],
                                            [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                                            [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                                            [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                                            [1.0, 2.0, 0.5],
                                            [1.0, 2.0, 0.5]],
                    two_boxes_for_ar1=True,
                    steps=[8, 16, 32, 64, 100, 300],
                    offsets=[0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                    clip_boxes=False,
                    variances=[0.1, 0.1, 0.2, 0.2],
                    normalize_coords=True,
                    subtract_mean=[123, 117, 104],
                    swap_channels=[2, 1, 0],
                    confidence_thresh=0.5,
                    iou_threshold=0.45,
                    top_k=200,
                    nms_max_output_size=400)
else:
    model = ssd_300(image_size=(img_height, img_width, 3),
                    n_classes=3,
                    mode='inference',
                    l2_regularization=0.0005,
                    scales=[0.1, 0.2, 0.37, 0.54, 0.71, 0.88, 1.05], # The scales for MS COCO are [0.07, 0.15, 0.33, 0.51, 0.69, 0.87, 1.05]
                    aspect_ratios_per_layer=[[1.0, 2.0, 0.5],
                                            [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                                            [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                                            [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                                            [1.0, 2.0, 0.5],
                                            [1.0, 2.0, 0.5]],
                    two_boxes_for_ar1=True,
                    steps=[8, 16, 32, 64, 100, 300],
                    offsets=[0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                    clip_boxes=False,
                    variances=[0.1, 0.1, 0.2, 0.2],
                    normalize_coords=True,
                    subtract_mean=[123, 117, 104],
                    swap_channels=[2, 1, 0],
                    confidence_thresh=0.5,
                    iou_threshold=0.45,
                    top_k=200,
                    nms_max_output_size=400)

# 2: Load the trained weights into the model.

# TODO: Set the path of the trained weights.
if args.dct:
    print("Loading the dct weights.")
    weights_path = '/tmp/ssd300_pascal_07+12_epoch-47_loss-4.2486_val_loss-4.9507.h5'
else:
    weights_path = '/save/2017018/bdegue01/experiments/ssd/RGB/ssd_miisst/ssd300_pascal_07+12_epoch-71_loss-1.5011_val_loss-2.8061.h5'

model.load_weights(weights_path)

# 3: Compile the model so that Keras won't complain the next time you load it.

adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
sgd = SGD(lr=0.001, momentum=0.9, decay=0.0, nesterov=False)

ssd_loss = SSDLoss(neg_pos_ratio=3, alpha=1.0)

model.compile(optimizer=sgd, loss=ssd_loss.compute_loss)

orig_images = [] # Store the images here.
input_images = [] # Store resized versions of the images here.

# We'll only load one image in this example.
img_path = '/save/2017018/bdegue01/datasets/VOC2007_test/JPEGImages/000010.jpg'

orig_images.append(imread(img_path))

# For the validation generator:
convert_to_3_channels = ConvertTo3Channels()
resize = Resize(height=300, width=300)

with Image.open(img_path) as image:

    img = np.array(image, dtype=np.uint8)
    img, _ = convert_to_3_channels(img, [])
    img, _ = resize(img, [[1,1,1,1,1]])
    input_images.append(np.array(img))
input_images = np.array(input_images, dtype=np.float64)

if args.dct:
    decoder = jpegdecoder.decoder.JPEGDecoder()

    im = Image.fromarray(np.array(img))
    im.save(os.path.join(os.environ["LOCAL_WORK_DIR"], "1.jpg"), format="jpeg", subsampling=0)
    while True:
        try:
            img = decoder.decode_file(os.path.join(os.environ["LOCAL_WORK_DIR"], "1.jpg"), 2)
            rows, cols = img.get_component_shape(0)[0:2]
            if img.get_number_of_component() == 1:
                input_images[0, :, :, 0] = np.reshape(img.get_data(0), (rows, cols))[
                    :300, :300]
                input_images[0, :, :, 1] = input_images[0, :, :, 0]
                input_images[0, :, :, 2] = input_images[0, :, :, 0]
            else:
                input_images[0, :, :, 0] = np.reshape(
                    img.get_data(0), (rows, cols))[:300, :300]
                input_images[0, :, :, 1] = np.reshape(
                    img.get_data(1), (rows, cols))[:300, :300]
                input_images[0, :, :, 2] = np.reshape(
                    img.get_data(2), (rows, cols))[:300, :300]
        except OSError:
            print("i'am stuck 1.jpg")
            continue
        except Exception as e:
            print(e)
            print("1.jpg")
            continue
        break
model.summary()
y_pred = model.predict(input_images)
print(y_pred.shape)
print(y_pred)
confidence_threshold = 0.2

y_pred_thresh = [y_pred[k][y_pred[k,:,1] > confidence_threshold] for k in range(y_pred.shape[0])]
print(len(y_pred_thresh))
print(y_pred_thresh)
# np.set_printoptions(precision=2, suppress=True, linewidth=90)
# print("Predicted boxes:\n")
# print('   class   conf xmin   ymin   xmax   ymax')
# print(y_pred_thresh[0])

# Display the image and draw the predicted boxes onto it.

# Set the colors for the bounding boxes
colors = plt.cm.hsv(np.linspace(0, 1, 21)).tolist()
classes = ['background',
           'car', 'truck', 'motorcycle']
classes = ['background',
            'aeroplane', 'bicycle', 'bird', 'boat',
            'bottle', 'bus', 'car', 'cat',
            'chair', 'cow', 'diningtable', 'dog',
            'horse', 'motorbike', 'person', 'pottedplant',
            'sheep', 'sofa', 'train', 'tvmonitor']

plt.figure(figsize=(20,12))
plt.imshow(orig_images[0])

current_axis = plt.gca()


for box in y_pred_thresh[0]:
    # Transform the predicted bounding boxes for the 300x300 image to the original image dimensions.
    xmin = box[2] * orig_images[0].shape[1] / img_width
    ymin = box[3] * orig_images[0].shape[0] / img_height
    xmax = box[4] * orig_images[0].shape[1] / img_width
    ymax = box[5] * orig_images[0].shape[0] / img_height
    color = colors[int(box[0])]
    label = '{}: {:.2f}'.format(classes[int(box[0])], box[1])
    current_axis.add_patch(plt.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, color=color, fill=False, linewidth=2))  
    current_axis.text(xmin, ymin, label, size='x-large', color='white', bbox={'facecolor':color, 'alpha':1.0})

plt.show()
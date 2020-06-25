import os

from os.path import join
from os import environ

from argparse import ArgumentParser

parser = ArgumentParser(description="Script to train the SSD Resnet on the pascal voc dataset.")
parser.add_argument("--weights", default=None, help="The weights to load into the model")
parser.add_argument("-vd", "--visible_device", help="The device to use when training with the GPU", default="-1")
parser.add_argument("--restart", default=None, help="Wether the simulation starts from a previous save")

parser.add_argument("--archi", help="""The network architecture to use, value can be :\n
* cb5_only : CbCr and Y only go through the conv block 5 of Resnet50\n
* deconv : deconvolution architecture of Über article\n
* up_sampling : up sampling architecture of Über article\n
* y_cb4_cbcr_cb5 :Y go through the conv block 4 of Resnet50 and CbCr go through conv block 5\n
* ssd_custom : the extra-feature layers of SSD are removed to match dimension with full Late-concat-RFA architecture of Über
""")

loading_check = parser.add_mutually_exclusive_group(required=True)
loading_check.add_argument("--ssd", action="store_true")
loading_check.add_argument("--resnet", action="store_true")

ssd_augmentation = parser.add_mutually_exclusive_group(required=True)
ssd_augmentation.add_argument("--crop", action="store_true")
ssd_augmentation.add_argument("--no_crop", action="store_true")

training_set = parser.add_mutually_exclusive_group(required=True)
training_set.add_argument("--p07", action="store_true")
training_set.add_argument("--p07p12", action="store_true")

regularizer = parser.add_mutually_exclusive_group(required=True)
regularizer.add_argument("--reg", action="store_true")
regularizer.add_argument("--no_reg", action="store_true")

args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"]=args.visible_device
if "LOCAL_WORK_DIR" not in os.environ:
    os.environ["LOCAL_WORK_DIR"] = "./" + os.environ["CUDA_VISIBLE_DEVICES"]
else :
    os.environ["LOCAL_WORK_DIR"] = os.path.join(os.environ["LOCAL_WORK_DIR"], os.environ["CUDA_VISIBLE_DEVICES"])

if not os.path.exists(os.environ["LOCAL_WORK_DIR"]):
    os.mkdir(os.environ["LOCAL_WORK_DIR"])

if args.archi == "deconv":
    deconv = True
else:
    deconv = False

from keras.metrics import top_k_categorical_accuracy
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint, TerminateOnNaN, CSVLogger, ReduceLROnPlateau, EarlyStopping, TensorBoard
from keras_layers.keras_layer_AnchorBoxes import AnchorBoxes

from keras import backend as K
from keras.models import load_model
from math import ceil
import numpy as np
from shutil import copyfile

from keras.models import Model
from keras.utils import multi_gpu_model
from keras.models import load_model

from models.keras_ssd300_dct_j2d_resnet import ssd_resnet_EF_layers_identical, ssd_resnet_EF_layers_custom
from keras_loss_function.keras_ssd_loss import SSDLoss

from ssd_encoder_decoder.ssd_input_encoder import SSDInputEncoder

from data_generator.object_detection_2d_data_generator_dct_j2d import DataGeneratorDCT

from data_generator.object_detection_2d_geometric_ops import Resize
from data_generator.object_detection_2d_photometric_ops import ConvertTo3Channels

from data_generator.data_augmentation_chain_original_ssd import SSDDataAugmentation
from data_generator.data_augmentation_chain_original_ssd_no_crop import SSDDataAugmentationNoCrop


from keras_layers.keras_layer_L2Normalization import L2Normalization

def _top_k_accuracy(k):
    def _func(y_true, y_pred):
        return top_k_categorical_accuracy(y_true, y_pred, k)
    return _func

copyfile("training_dct_pascal_j2d_resnet.py", os.path.join(os.environ["LOCAL_WORK_DIR"], "training_dct_pascal_j2d_resnet.py"))
copyfile("models/keras_ssd300_dct_j2d_resnet.py", os.path.join(os.environ["LOCAL_WORK_DIR"], "keras_ssd300_dct_j2d_resnet.py"))

img_height = 300
img_width = 300
img_channels = 3

n_classes = 20
scales = [0.1, 0.2, 0.37, 0.54, 0.71, 0.88, 1.05]

aspect_ratios = [[1.0, 2.0, 0.5],
                 [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                 [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                 [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                 [1.0, 2.0, 0.5],
                 [1.0, 2.0, 0.5]]

two_boxes_for_ar1 = True
steps = [8, 16, 32, 64, 100, 300]
offsets = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
clip_boxes = False
variances = [0.1, 0.1, 0.2, 0.2]
normalize_coords = True

ssd_args = {"image_size":(img_height, img_width, img_channels),
            "n_classes":n_classes,
            "mode":'training',
            "l2_regularization":0.0005,
            "scales":scales,
            "aspect_ratios_per_layer":aspect_ratios,
            "two_boxes_for_ar1":two_boxes_for_ar1,
            "steps":steps,
            "offsets":offsets,
            "clip_boxes":clip_boxes,
            "variances":variances,
            "normalize_coords":normalize_coords,
            "archi":args.archi}

if args.archi == "ssd_custom":
    model = ssd_resnet_EF_layers_custom(**ssd_args)
else:
    model = ssd_resnet_EF_layers_identical(**ssd_args)

model_layers = []

for i in range(len(model.layers)):
    model_layers.append(model.layers[i].name)

if args.restart:
    model.load_weights(args.restart, by_name=True)
elif args.weights:
    if args.resnet:
        temp_model = load_model(args.weights, custom_objects={'_func': _top_k_accuracy(1), "_func_1": _top_k_accuracy(1)} )
    elif args.ssd:
        temp_model = load_model(args.weights, custom_objects={'AnchorBoxes': AnchorBoxes})
    else:
        raise Exception("Error, the type of check should have been specified.")

    temp_model.summary()

    model.load_weights(args.weights, by_name=True)

# 3: Instantiate an optimizer and the SSD loss function and compile the model.
sgd = SGD(lr=0.001, momentum=0.9, decay=0.0, nesterov=False)

ssd_loss = SSDLoss(neg_pos_ratio=3, alpha=1.0)

model.compile(optimizer=sgd, loss=ssd_loss.compute_loss)

# 1: Instantiate two `DataGenerator` objects: One for training, one for validation
train_dataset = DataGeneratorDCT(load_images_into_memory=False, hdf5_dataset_path=None)
val_dataset = DataGeneratorDCT(load_images_into_memory=False, hdf5_dataset_path=None)

# 2: Parse the image and label lists for the training and validation datasets. This can take a while.
VOC_2007_images_dir = join(environ['DATASET_PATH'], 'VOC2007/JPEGImages/')
VOC_2007_test_images_dir = join(environ['DATASET_PATH'], 'VOC2007_test/JPEGImages/')
VOC_2012_images_dir = join(environ['DATASET_PATH'], 'VOC2012/JPEGImages/')

# The directories that contain the annotations.
VOC_2007_annotations_dir = join(environ['DATASET_PATH'], 'VOC2007/Annotations/')
VOC_2007_test_annotations_dir = join(environ['DATASET_PATH'], 'VOC2007_test/Annotations/')
VOC_2012_annotations_dir = join(environ['DATASET_PATH'], 'VOC2012/Annotations/')

# The paths to the image sets.
VOC_2007_train_image_set_filename = join(environ['DATASET_PATH'], 'VOC2007/ImageSets/Main/train.txt')
VOC_2012_train_image_set_filename = join(environ['DATASET_PATH'], 'VOC2012/ImageSets/Main/train.txt')
VOC_2007_val_image_set_filename = join(environ['DATASET_PATH'], 'VOC2007/ImageSets/Main/val.txt')
VOC_2012_val_image_set_filename = join(environ['DATASET_PATH'], 'VOC2012/ImageSets/Main/val.txt')
VOC_2007_trainval_image_set_filename = join(environ['DATASET_PATH'], 'VOC2007/ImageSets/Main/trainval.txt')
VOC_2012_trainval_image_set_filename = join(environ['DATASET_PATH'], 'VOC2012/ImageSets/Main/trainval.txt')
VOC_2007_test_image_set_filename = join(environ['DATASET_PATH'], 'VOC2007_test/ImageSets/Main/test.txt')

# The XML parser needs to now what object class names to look for and in which order to map them to integers.
classes = ['background',
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat',
           'chair', 'cow', 'diningtable', 'dog',
           'horse', 'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor']

if args.p07:
    train_images_dirs = [VOC_2007_images_dir]
    train_image_set_filenames = [VOC_2007_train_image_set_filename]
    train_annotation_dirs = [VOC_2007_annotations_dir]
    val_images_dirs = [VOC_2007_images_dir]
    val_image_set_filenames = [VOC_2007_val_image_set_filename]
    val_annotation_dirs = [VOC_2007_annotations_dir]
elif args.p07p12:
    train_images_dirs = [VOC_2007_images_dir, VOC_2012_images_dir]
    train_image_set_filenames = [VOC_2007_train_image_set_filename, VOC_2012_train_image_set_filename]
    train_annotation_dirs = [VOC_2007_annotations_dir, VOC_2012_annotations_dir]
    val_images_dirs = [VOC_2007_images_dir, VOC_2012_images_dir]
    val_image_set_filenames = [VOC_2007_val_image_set_filename, VOC_2012_val_image_set_filename]
    val_annotation_dirs = [VOC_2007_annotations_dir, VOC_2012_annotations_dir]

else:
    raise Exception("Dataset not supported, I don't now how you got here.")

train_dataset.parse_xml(images_dirs=train_images_dirs,
                        image_set_filenames=train_image_set_filenames,
                        annotations_dirs=train_annotation_dirs,
                        classes=classes,
                        include_classes='all',
                        exclude_truncated=False,
                        exclude_difficult=False,
                        ret=False)

val_dataset.parse_xml(images_dirs=val_images_dirs,
                      image_set_filenames=val_image_set_filenames,
                      annotations_dirs=val_annotation_dirs,
                      classes=classes,
                      include_classes='all',
                      exclude_truncated=False,
                      exclude_difficult=True,
                      ret=False)


# 3: Set the batch size.
batch_size = 32

# 4: Set the image transformations for pre-processing and data augmentation options.
if args.crop:
    ssd_data_augmentation = SSDDataAugmentation(img_height=img_height,
                                                img_width=img_width,
                                                background=[123, 117, 104])
elif args.no_crop:
    ssd_data_augmentation = SSDDataAugmentationNoCrop(img_height=img_height,
                                                img_width=img_width,
                                                background=[0, 0, 0])

# For the validation generator:
convert_to_3_channels = ConvertTo3Channels()
resize = Resize(height=img_height, width=img_width)

# 5: Instantiate an encoder that can encode ground truth labels into the format needed by the SSD loss function.
predictor_sizes = [model.get_layer('conv4_3_norm_mbox_conf_{}'.format(n_classes+1)).output_shape[1:3],
                   model.get_layer('fc7_mbox_conf_{}'.format(n_classes+1)).output_shape[1:3],
                   model.get_layer('conv6_2_mbox_conf_{}'.format(n_classes+1)).output_shape[1:3],
                   model.get_layer('conv7_2_mbox_conf_{}'.format(n_classes+1)).output_shape[1:3],
                   model.get_layer('conv8_2_mbox_conf_{}'.format(n_classes+1)).output_shape[1:3],
                   model.get_layer('conv9_2_mbox_conf_{}'.format(n_classes+1)).output_shape[1:3]]

ssd_input_encoder = SSDInputEncoder(img_height=img_height,
                                    img_width=img_width,
                                    n_classes=n_classes,
                                    predictor_sizes=predictor_sizes,
                                    scales=scales,
                                    aspect_ratios_per_layer=aspect_ratios,
                                    two_boxes_for_ar1=two_boxes_for_ar1,
                                    steps=steps,
                                    offsets=offsets,
                                    clip_boxes=clip_boxes,
                                    variances=variances,
                                    matching_type='multi',
                                    pos_iou_threshold=0.5,
                                    neg_iou_limit=0.5,
                                    normalize_coords=normalize_coords)

# 6: Create the generator handles that will be passed to Keras' `fit_generator()` function.

train_generator = train_dataset.generate(batch_size=batch_size,
                                         shuffle=True,
                                         transformations=[ssd_data_augmentation],
                                         label_encoder=ssd_input_encoder,
                                         returns={'processed_images',
                                                  'encoded_labels'},
                                         keep_images_without_gt=False, deconv=deconv)

val_generator = val_dataset.generate(batch_size=batch_size,
                                     shuffle=False,
                                     transformations=[convert_to_3_channels,
                                                      resize],
                                     label_encoder=ssd_input_encoder,
                                     returns={'processed_images',
                                              'encoded_labels'},
                                     keep_images_without_gt=False, deconv=deconv)

# Get the number of samples in the training and validations datasets.
train_dataset_size = train_dataset.get_dataset_size()
val_dataset_size   = val_dataset.get_dataset_size()

print("Number of images in the training dataset:\t{:>6}".format(train_dataset_size))
print("Number of images in the validation dataset:\t{:>6}".format(val_dataset_size))

# Define a learning rate schedule.

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=7)
early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=10)
tensorboard = TensorBoard(log_dir=os.path.join(os.environ["LOCAL_WORK_DIR"],'./logs'))

# Define model callbacks.

model_checkpoint = ModelCheckpoint(filepath=os.path.join(os.environ["EXPERIMENTS_OUTPUT_DIRECTORY"], 'ssd300_pascal_07+12_epoch-{epoch:02d}_loss-{loss:.4f}_val_loss-{val_loss:.4f}.h5'),
                                   monitor='val_loss',
                                   verbose=1,
                                   save_best_only=True,
                                   save_weights_only=False,
                                   mode='auto',
                                   period=1)

csv_logger = CSVLogger(filename=os.path.join(os.environ["EXPERIMENTS_OUTPUT_DIRECTORY"], 'ssd300_pascal_07+12_training_log.csv'),
                       separator=',',
                       append=True)

terminate_on_nan = TerminateOnNaN()

callbacks = [model_checkpoint,
             csv_logger,
             reduce_lr,
             terminate_on_nan,
             tensorboard,
             early_stop]

# If you're resuming a previous training, set `initial_epoch` and `final_epoch` accordingly.
if args.restart:
    initial_epoch = int(args.restart.split("-")[1].split("_")[0])
else:
    initial_epoch   = 0
final_epoch     = 480
steps_per_epoch = 1000

history = model.fit_generator(generator=train_generator,
                              steps_per_epoch=steps_per_epoch,
                              epochs=final_epoch,
                              callbacks=callbacks,
                              validation_data=val_generator,
                              validation_steps=ceil(val_dataset_size/batch_size),
                              initial_epoch=initial_epoch)

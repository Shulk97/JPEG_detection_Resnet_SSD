"""
A Keras port of the original Caffe SSD300 network.

Copyright (C) 2018 Pierluigi Ferrari

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from __future__ import division
import numpy as np
from keras.models import Model
from keras.layers import (
    GlobalAveragePooling2D,
    Add,
    Input,
    Lambda,
    Activation,
    Conv2D,
    MaxPooling2D,
    ZeroPadding2D,
    Reshape,
    Concatenate,
    BatchNormalization,
    UpSampling2D,
    Conv2DTranspose
)
from keras.regularizers import l2
from keras import backend as K

from keras_layers.keras_layer_AnchorBoxes import AnchorBoxes
from keras_layers.keras_layer_L2Normalization import L2Normalization
from keras_layers.keras_layer_DecodeDetections import DecodeDetections
from keras_layers.keras_layer_DecodeDetectionsFast import DecodeDetectionsFast


def identity_block(input_tensor, kernel_size, filters, stage, block):
    """The identity block is the block that has no conv layer at shortcut.

    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of
            middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names

    # Returns
        Output tensor for the block.
    """
    filters1, filters2, filters3 = filters
    bn_axis = 3
    conv_name_base = "res" + str(stage) + block + "_branch"
    bn_name_base = "bn" + str(stage) + block + "_branch"

    # res4f_branch2c
    # bn4f_branch2c

    # filters: Integer, the dimensionality of the output space (i.e. the number of output filters in the convolution).
    # kernel_size: An integer or tuple/list of 2 integers, specifying the height and width of the 2D convolution window.
    # Can be a single integer to specify the same value for all spatial dimensions.

    # keras.layers.Conv2D(filters, kernel_size, strides=(1, 1), padding='valid', data_format=None, dilation_rate=(1, 1), activation=None,
    # use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None,
    # activity_regularizer=None, kernel_constraint=None, bias_constraint=None)
    # padding="valid" -> no padding

    x = Conv2D(
        filters1, (1, 1), kernel_initializer="he_normal", name=conv_name_base + "2a"
    )(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + "2a")(x)
    x = Activation("relu")(x)

    x = Conv2D(
        filters2,
        kernel_size,
        padding="same",
        kernel_initializer="he_normal",
        name=conv_name_base + "2b",
    )(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + "2b")(x)
    x = Activation("relu")(x)

    x = Conv2D(
        filters3, (1, 1), kernel_initializer="he_normal", name=conv_name_base + "2c"
    )(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + "2c")(x)

    x = Add()([x, input_tensor])
    x = Activation("relu")(x)
    return x


def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):
    """A block that has a conv layer at shortcut.

    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of
            middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
        strides: Strides for the first conv layer in the block.

    # Returns
        Output tensor for the block.

    Note that from stage 3,
    the first conv layer at main path is with strides=(2, 2)
    And the shortcut should have strides=(2, 2) as well
    """
    filters1, filters2, filters3 = filters
    bn_axis = 3

    conv_name_base = "res" + str(stage) + block + "_branch"
    bn_name_base = "bn" + str(stage) + block + "_branch"

    x = Conv2D(
        filters1,
        (1, 1),
        strides=strides,
        kernel_initializer="he_normal",
        name=conv_name_base + "2a",
    )(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + "2a")(x)
    x = Activation("relu")(x)

    x = Conv2D(
        filters2,
        kernel_size,
        padding="same",
        kernel_initializer="he_normal",
        name=conv_name_base + "2b",
    )(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + "2b")(x)
    x = Activation("relu")(x)

    x = Conv2D(
        filters3, (1, 1), kernel_initializer="he_normal", name=conv_name_base + "2c"
    )(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + "2c")(x)

    shortcut = Conv2D(
        filters3,
        (1, 1),
        strides=strides,
        kernel_initializer="he_normal",
        name=conv_name_base + "1",
    )(input_tensor)
    shortcut = BatchNormalization(axis=bn_axis, name=bn_name_base + "1")(shortcut)

    x = Add()([x, shortcut])
    x = Activation("relu")(x)
    return x


def ssd_resnet_EF_layers_custom(
    image_size,
    n_classes,
    mode="training",
    l2_regularization=0.0005,
    min_scale=None,
    max_scale=None,
    scales=None,
    aspect_ratios_global=None,
    aspect_ratios_per_layer=[
        [1.0, 2.0, 0.5],
        [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
        [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
        [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
        [1.0, 2.0, 0.5],
        [1.0, 2.0, 0.5],
    ],
    two_boxes_for_ar1=True,
    steps=[8, 16, 32, 64, 100, 300],
    offsets=None,
    clip_boxes=False,
    variances=[0.1, 0.1, 0.2, 0.2],
    coords="centroids",
    normalize_coords=True,
    subtract_mean=[123, 117, 104],
    divide_by_stddev=None,
    swap_channels=[2, 1, 0],
    confidence_thresh=0.01,
    iou_threshold=0.45,
    top_k=200,
    nms_max_output_size=400,
    return_predictor_sizes=False,
    archi="ssd_custom"
):
    """
    Build a Keras model with SSD300 architecture, see references.

    The base network is a reduced atrous VGG-16, extended by the SSD architecture,
    as described in the paper.

    Most of the arguments that this function takes are only needed for the anchor
    box layers. In case you're training the network, the parameters passed here must
    be the same as the ones used to set up `SSDBoxEncoder`. In case you're loading
    trained weights, the parameters passed here must be the same as the ones used
    to produce the trained weights.

    Some of these arguments are explained in more detail in the documentation of the
    `SSDBoxEncoder` class.

    Note: Requires Keras v2.0 or later. Currently works only with the
    TensorFlow backend (v1.0 or later).

    Arguments:
        image_size (tuple): The input image size in the format `(height, width, channels)`.
        n_classes (int): The number of positive classes, e.g. 20 for Pascal VOC, 80 for MS COCO.
        mode (str, optional): One of 'training', 'inference' and 'inference_fast'. In 'training' mode,
            the model outputs the raw prediction tensor, while in 'inference' and 'inference_fast' modes,
            the raw predictions are decoded into absolute coordinates and filtered via confidence thresholding,
            non-maximum suppression, and top-k filtering. The difference between latter two modes is that
            'inference' follows the exact procedure of the original Caffe implementation, while
            'inference_fast' uses a faster prediction decoding procedure.
        l2_regularization (float, optional): The L2-regularization rate. Applies to all convolutional layers.
            Set to zero to deactivate L2-regularization.
        min_scale (float, optional): The smallest scaling factor for the size of the anchor boxes as a fraction
            of the shorter side of the input images.
        max_scale (float, optional): The largest scaling factor for the size of the anchor boxes as a fraction
            of the shorter side of the input images. All scaling factors between the smallest and the
            largest will be linearly interpolated. Note that the second to last of the linearly interpolated
            scaling factors will actually be the scaling factor for the last predictor layer, while the last
            scaling factor is used for the second box for aspect ratio 1 in the last predictor layer
            if `two_boxes_for_ar1` is `True`.
        scales (list, optional): A list of floats containing scaling factors per convolutional predictor layer.
            This list must be one element longer than the number of predictor layers. The first `k` elements are the
            scaling factors for the `k` predictor layers, while the last element is used for the second box
            for aspect ratio 1 in the last predictor layer if `two_boxes_for_ar1` is `True`. This additional
            last scaling factor must be passed either way, even if it is not being used. If a list is passed,
            this argument overrides `min_scale` and `max_scale`. All scaling factors must be greater than zero.
        aspect_ratios_global (list, optional): The list of aspect ratios for which anchor boxes are to be
            generated. This list is valid for all prediction layers.
        aspect_ratios_per_layer (list, optional): A list containing one aspect ratio list for each prediction layer.
            This allows you to set the aspect ratios for each predictor layer individually, which is the case for the
            original SSD300 implementation. If a list is passed, it overrides `aspect_ratios_global`.
        two_boxes_for_ar1 (bool, optional): Only relevant for aspect ratio lists that contain 1. Will be ignored otherwise.
            If `True`, two anchor boxes will be generated for aspect ratio 1. The first will be generated
            using the scaling factor for the respective layer, the second one will be generated using
            geometric mean of said scaling factor and next bigger scaling factor.
        steps (list, optional): `None` or a list with as many elements as there are predictor layers. The elements can be
            either ints/floats or tuples of two ints/floats. These numbers represent for each predictor layer how many
            pixels apart the anchor box center points should be vertically and horizontally along the spatial grid over
            the image. If the list contains ints/floats, then that value will be used for both spatial dimensions.
            If the list contains tuples of two ints/floats, then they represent `(step_height, step_width)`.
            If no steps are provided, then they will be computed such that the anchor box center points will form an
            equidistant grid within the image dimensions.
        offsets (list, optional): `None` or a list with as many elements as there are predictor layers. The elements can be
            either floats or tuples of two floats. These numbers represent for each predictor layer how many
            pixels from the top and left boarders of the image the top-most and left-most anchor box center points should be
            as a fraction of `steps`. The last bit is important: The offsets are not absolute pixel values, but fractions
            of the step size specified in the `steps` argument. If the list contains floats, then that value will
            be used for both spatial dimensions. If the list contains tuples of two floats, then they represent
            `(vertical_offset, horizontal_offset)`. If no offsets are provided, then they will default to 0.5 of the step size.
        clip_boxes (bool, optional): If `True`, clips the anchor box coordinates to stay within image boundaries.
        variances (list, optional): A list of 4 floats >0. The anchor box offset for each coordinate will be divided by
            its respective variance value.
        coords (str, optional): The box coordinate format to be used internally by the model (i.e. this is not the input format
            of the ground truth labels). Can be either 'centroids' for the format `(cx, cy, w, h)` (box center coordinates, width,
            and height), 'minmax' for the format `(xmin, xmax, ymin, ymax)`, or 'corners' for the format `(xmin, ymin, xmax, ymax)`.
        normalize_coords (bool, optional): Set to `True` if the model is supposed to use relative instead of absolute coordinates,
            i.e. if the model predicts box coordinates within [0,1] instead of absolute coordinates.
        subtract_mean (array-like, optional): `None` or an array-like object of integers or floating point values
            of any shape that is broadcast-compatible with the image shape. The elements of this array will be
            subtracted from the image pixel intensity values. For example, pass a list of three integers
            to perform per-channel mean normalization for color images.
        divide_by_stddev (array-like, optional): `None` or an array-like object of non-zero integers or
            floating point values of any shape that is broadcast-compatible with the image shape. The image pixel
            intensity values will be divided by the elements of this array. For example, pass a list
            of three integers to perform per-channel standard deviation normalization for color images.
        swap_channels (list, optional): Either `False` or a list of integers representing the desired order in which the input
            image channels should be swapped.
        confidence_thresh (float, optional): A float in [0,1), the minimum classification confidence in a specific
            positive class in order to be considered for the non-maximum suppression stage for the respective class.
            A lower value will result in a larger part of the selection process being done by the non-maximum suppression
            stage, while a larger value will result in a larger part of the selection process happening in the confidence
            thresholding stage.
        iou_threshold (float, optional): A float in [0,1]. All boxes that have a Jaccard similarity of greater than `iou_threshold`
            with a locally maximal box will be removed from the set of predictions for a given class, where 'maximal' refers
            to the box's confidence score.
        top_k (int, optional): The number of highest scoring predictions to be kept for each batch item after the
            non-maximum suppression stage.
        nms_max_output_size (int, optional): The maximal number of predictions that will be left over after the NMS stage.
        return_predictor_sizes (bool, optional): If `True`, this function not only returns the model, but also
            a list containing the spatial dimensions of the predictor layers. This isn't strictly necessary since
            you can always get their sizes easily via the Keras API, but it's convenient and less error-prone
            to get them this way. They are only relevant for training anyway (SSDBoxEncoder needs to know the
            spatial dimensions of the predictor layers), for inference you don't need them.

    Returns:
        model: The Keras SSD300 model.
        predictor_sizes (optional): A Numpy array containing the `(height, width)` portion
            of the output tensor shape for each convolutional predictor layer. During
            training, the generator function needs this in order to transform
            the ground truth labels into tensors of identical structure as the
            output tensors of the model, which is in turn needed for the cost
            function.

    References:
        https://arxiv.org/abs/1512.02325v5
    """

    n_predictor_layers = 6  # The number of predictor conv layers in the network is 6 for the original SSD300.
    n_classes += 1  # Account for the background class.
    l2_reg = l2_regularization  # Make the internal name shorter.
    img_height, img_width, img_channels = image_size[0], image_size[1], image_size[2]

    ############################################################################
    # Get a few exceptions out of the way.
    ############################################################################

    if aspect_ratios_global is None and aspect_ratios_per_layer is None:
        raise ValueError(
            "`aspect_ratios_global` and `aspect_ratios_per_layer` cannot both be None. At least one needs to be specified."
        )
    if aspect_ratios_per_layer:
        if len(aspect_ratios_per_layer) != n_predictor_layers:
            raise ValueError(
                "It must be either aspect_ratios_per_layer is None or len(aspect_ratios_per_layer) == {}, but len(aspect_ratios_per_layer) == {}.".format(
                    n_predictor_layers, len(aspect_ratios_per_layer)
                )
            )

    if (min_scale is None or max_scale is None) and scales is None:
        raise ValueError(
            "Either `min_scale` and `max_scale` or `scales` need to be specified."
        )
    if scales:
        if len(scales) != n_predictor_layers + 1:
            raise ValueError(
                "It must be either scales is None or len(scales) == {}, but len(scales) == {}.".format(
                    n_predictor_layers + 1, len(scales)
                )
            )
    else:  # If no explicit list of scaling factors was passed, compute the list of scaling factors from `min_scale` and `max_scale`
        scales = np.linspace(min_scale, max_scale, n_predictor_layers + 1)

    if len(variances) != 4:
        raise ValueError(
            "4 variance values must be pased, but {} values were received.".format(
                len(variances)
            )
        )
    variances = np.array(variances)
    if np.any(variances <= 0):
        raise ValueError(
            "All variances must be >0, but the variances given are {}".format(variances)
        )

    if (not (steps is None)) and (len(steps) != n_predictor_layers):
        raise ValueError(
            "You must provide at least one step value per predictor layer."
        )

    if (not (offsets is None)) and (len(offsets) != n_predictor_layers):
        raise ValueError(
            "You must provide at least one offset value per predictor layer."
        )

    ############################################################################
    # Compute the anchor box parameters.
    ############################################################################

    # Set the aspect ratios for each predictor layer. These are only needed for the anchor box layers.
    if aspect_ratios_per_layer:
        aspect_ratios = aspect_ratios_per_layer
    else:
        aspect_ratios = [aspect_ratios_global] * n_predictor_layers

    # Compute the number of boxes to be predicted per cell for each predictor layer.
    # We need this so that we know how many channels the predictor layers need to have.
    if aspect_ratios_per_layer:
        n_boxes = []
        for ar in aspect_ratios_per_layer:
            if (1 in ar) & two_boxes_for_ar1:
                n_boxes.append(len(ar) + 1)  # +1 for the second box for aspect ratio 1
            else:
                n_boxes.append(len(ar))
    else:  # If only a global aspect ratio list was passed, then the number of boxes is the same for each predictor layer
        if (1 in aspect_ratios_global) & two_boxes_for_ar1:
            n_boxes = len(aspect_ratios_global) + 1
        else:
            n_boxes = len(aspect_ratios_global)
        n_boxes = [n_boxes] * n_predictor_layers

    if steps is None:
        steps = [None] * n_predictor_layers
    if offsets is None:
        offsets = [None] * n_predictor_layers

    ############################################################################
    # Define functions for the Lambda layers below.
    ############################################################################

    def identity_layer(tensor):
        return tensor

    def input_mean_normalization(tensor):
        return tensor - np.array(subtract_mean)

    def input_stddev_normalization(tensor):
        return tensor / np.array(divide_by_stddev)

    def input_channel_swap(tensor):
        if len(swap_channels) == 3:
            return K.stack(
                [
                    tensor[..., swap_channels[0]],
                    tensor[..., swap_channels[1]],
                    tensor[..., swap_channels[2]],
                ],
                axis=-1,
            )
        elif len(swap_channels) == 4:
            return K.stack(
                [
                    tensor[..., swap_channels[0]],
                    tensor[..., swap_channels[1]],
                    tensor[..., swap_channels[2]],
                    tensor[..., swap_channels[3]],
                ],
                axis=-1,
            )

    ############################################################################
    # Build the network.
    ############################################################################
    input_shape_y = (38, 38, 64)
    input_shape_cbcr = (19, 19, 128)

    input_y = Input(input_shape_y)
    input_cbcr = Input(input_shape_cbcr)

    y = BatchNormalization(input_shape=input_shape_y)(input_y)
    y = conv_block(y, 1, [256, 256, 384], stage=1, block="a2", strides=(1, 1))
    y = identity_block(y, 2, [256, 256, 384], stage=1, block="b2")
    y = identity_block(y, 3, [256, 256, 384], stage=1, block="c2")

    y = conv_block(y, 3, [128, 128, 384], stage=2, block="a3", strides=(1, 1))
    y = identity_block(y, 3, [128, 128, 384], stage=2, block="b3")
    y = identity_block(y, 3, [128, 128, 384], stage=2, block="c3")
    conv4_3 = identity_block(y, 3, [128, 128, 384], stage=2, block="d3")
    
    y = conv_block(conv4_3, 3, [256, 256, 384], stage=2, block="a4")
    
    cbcr = BatchNormalization(input_shape=input_shape_cbcr)(input_cbcr)
    cbcr = conv_block(cbcr, 1, [256, 256, 128], stage=2, block="a5", strides=(1, 1))
    
    x = Concatenate(axis=-1)([y, cbcr])

    # Block 3
    x = identity_block(x, 3, [128, 128, 512], stage=3, block="b")
    x = identity_block(x, 3, [128, 128, 512], stage=3, block="c")
    conv3_3 = identity_block(x, 3, [128, 128, 512], stage=3, block="d")

    # Block 4
    x = conv_block(conv3_3, 3, [256, 256, 1024], stage=4, block="a")
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block="b")
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block="c")
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block="d")
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block="e")
    conv4_6 = identity_block(x, 3, [256, 256, 1024], stage=4, block="f")

    # Block 5
    x = conv_block(conv4_6, 3, [512, 512, 2048], stage=5, block="a")
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block="b")
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block="c")

    pool5 = MaxPooling2D((3, 3), strides=(1, 1), padding="same", name="pool5_ssd")(x)

    fc6 = Conv2D(
        1024,
        (3, 3),
        dilation_rate=(6, 6),
        activation="relu",
        padding="same",
        kernel_initializer="he_normal",
        kernel_regularizer=l2(l2_reg),
        name="fc6",
    )(pool5)

    fc7 = Conv2D(
        1024,
        (1, 1),
        activation="relu",
        padding="same",
        kernel_initializer="he_normal",
        kernel_regularizer=l2(l2_reg),
        name="fc7",
    )(fc6)

    conv6_1 = Conv2D(
        256,
        (1, 1),
        activation="relu",
        padding="same",
        kernel_initializer="he_normal",
        kernel_regularizer=l2(l2_reg),
        name="conv6_1",
    )(fc7)

    conv6_1 = ZeroPadding2D(padding=((1, 1), (1, 1)), name="conv6_padding")(conv6_1)
    
    conv6_2 = Conv2D(
        256,
        (3, 3),
        strides=(2, 2),
        activation="relu",
        padding="valid",
        kernel_initializer="he_normal",
        kernel_regularizer=l2(l2_reg),
        name="conv6_2",
    )(conv6_1)
    
    conv9_1 = Conv2D(
        128,
        (1, 1),
        activation="relu",
        padding="same",
        kernel_initializer="he_normal",
        kernel_regularizer=l2(l2_reg),
        name="conv9_1",
    )(conv6_2)
    conv9_2 = Conv2D(
        256,
        (3, 3),
        strides=(1, 1),
        activation="relu",
        padding="valid",
        kernel_initializer="he_normal",
        kernel_regularizer=l2(l2_reg),
        name="conv9_2",
    )(conv9_1)

    conv4_3_norm = L2Normalization(gamma_init=20, name="conv4_3_norm")(conv4_3)
    conv3_3_norm = L2Normalization(gamma_init=20, name="conv3_3_norm")(conv3_3)
    conv4_6_norm = L2Normalization(gamma_init=20, name="conv4_6_norm")(conv4_6)

    # Relation between old and new extra feature layers :
    # conv4_3 stays conv4_3 (VGG notation)
    # fc7 becomes conv3_3 (resnet)
    # conv6_2 becomes conv4_6 (resnet)
    # conv7_2 becomes fc7 (Ssd notation)
    # conv8_2 becomes conv6_2 (Ssd notation)

    ### Build the convolutional predictor layers on top of the base network

    # We precidt `n_classes` confidence values for each box, hence the confidence predictors have depth `n_boxes * n_classes`
    # Output shape of the confidence layers: `(batch, height, width, n_boxes * n_classes)`
    conv4_3_norm_mbox_conf = Conv2D(
        n_boxes[0] * n_classes,
        (3, 3),
        padding="same",
        kernel_initializer="he_normal",
        kernel_regularizer=l2(l2_reg),
        name="conv4_3_norm_mbox_conf_{}".format(n_classes),
    )(conv4_3_norm)
    fc7_mbox_conf = Conv2D(
        n_boxes[1] * n_classes,
        (3, 3),
        padding="same",
        kernel_initializer="he_normal",
        kernel_regularizer=l2(l2_reg),
        name="fc7_mbox_conf_{}".format(n_classes),
    )(
        conv3_3_norm
    )  # (fc7)
    conv6_2_mbox_conf = Conv2D(
        n_boxes[2] * n_classes,
        (3, 3),
        padding="same",
        kernel_initializer="he_normal",
        kernel_regularizer=l2(l2_reg),
        name="conv6_2_mbox_conf_{}".format(n_classes),
    )(
        conv4_6_norm
    )  # (conv6_2)
    conv7_2_mbox_conf = Conv2D(
        n_boxes[3] * n_classes,
        (3, 3),
        padding="same",
        kernel_initializer="he_normal",
        kernel_regularizer=l2(l2_reg),
        name="conv7_2_mbox_conf_{}".format(n_classes),
    )(
        fc7
    )  # (conv7_2)
    conv8_2_mbox_conf = Conv2D(
        n_boxes[4] * n_classes,
        (3, 3),
        padding="same",
        kernel_initializer="he_normal",
        kernel_regularizer=l2(l2_reg),
        name="conv8_2_mbox_conf_{}".format(n_classes),
    )(
        conv6_2
    )  # (conv8_2)
    conv9_2_mbox_conf = Conv2D(
        n_boxes[5] * n_classes,
        (3, 3),
        padding="same",
        kernel_initializer="he_normal",
        kernel_regularizer=l2(l2_reg),
        name="conv9_2_mbox_conf_{}".format(n_classes),
    )(conv9_2)
    # We predict 4 box coordinates for each box, hence the localization predictors have depth `n_boxes * 4`
    # Output shape of the localization layers: `(batch, height, width, n_boxes * 4)`
    conv4_3_norm_mbox_loc = Conv2D(
        n_boxes[0] * 4,
        (3, 3),
        padding="same",
        kernel_initializer="he_normal",
        kernel_regularizer=l2(l2_reg),
        name="conv4_3_norm_mbox_loc",
    )(conv4_3_norm)
    fc7_mbox_loc = Conv2D(
        n_boxes[1] * 4,
        (3, 3),
        padding="same",
        kernel_initializer="he_normal",
        kernel_regularizer=l2(l2_reg),
        name="fc7_mbox_loc",
    )(
        conv3_3_norm
    )  # (fc7)
    conv6_2_mbox_loc = Conv2D(
        n_boxes[2] * 4,
        (3, 3),
        padding="same",
        kernel_initializer="he_normal",
        kernel_regularizer=l2(l2_reg),
        name="conv6_2_mbox_loc",
    )(
        conv4_6_norm
    )  # (conv6_2)
    conv7_2_mbox_loc = Conv2D(
        n_boxes[3] * 4,
        (3, 3),
        padding="same",
        kernel_initializer="he_normal",
        kernel_regularizer=l2(l2_reg),
        name="conv7_2_mbox_loc",
    )(
        fc7
    )  # (conv7_2)
    conv8_2_mbox_loc = Conv2D(
        n_boxes[4] * 4,
        (3, 3),
        padding="same",
        kernel_initializer="he_normal",
        kernel_regularizer=l2(l2_reg),
        name="conv8_2_mbox_loc",
    )(
        conv6_2
    )  # (conv8_2)
    conv9_2_mbox_loc = Conv2D(
        n_boxes[5] * 4,
        (3, 3),
        padding="same",
        kernel_initializer="he_normal",
        kernel_regularizer=l2(l2_reg),
        name="conv9_2_mbox_loc",
    )(conv9_2)

    ### Generate the anchor boxes (called "priors" in the original Caffe/C++ implementation, so I'll keep their layer names)

    # Output shape of anchors: `(batch, height, width, n_boxes, 8)`
    conv4_3_norm_mbox_priorbox = AnchorBoxes(
        img_height,
        img_width,
        this_scale=scales[0],
        next_scale=scales[1],
        aspect_ratios=aspect_ratios[0],
        two_boxes_for_ar1=two_boxes_for_ar1,
        this_steps=steps[0],
        this_offsets=offsets[0],
        clip_boxes=clip_boxes,
        variances=variances,
        coords=coords,
        normalize_coords=normalize_coords,
        name="conv4_3_norm_mbox_priorbox",
    )(conv4_3_norm_mbox_loc)
    fc7_mbox_priorbox = AnchorBoxes(
        img_height,
        img_width,
        this_scale=scales[1],
        next_scale=scales[2],
        aspect_ratios=aspect_ratios[1],
        two_boxes_for_ar1=two_boxes_for_ar1,
        this_steps=steps[1],
        this_offsets=offsets[1],
        clip_boxes=clip_boxes,
        variances=variances,
        coords=coords,
        normalize_coords=normalize_coords,
        name="fc7_mbox_priorbox",
    )(fc7_mbox_loc)
    conv6_2_mbox_priorbox = AnchorBoxes(
        img_height,
        img_width,
        this_scale=scales[2],
        next_scale=scales[3],
        aspect_ratios=aspect_ratios[2],
        two_boxes_for_ar1=two_boxes_for_ar1,
        this_steps=steps[2],
        this_offsets=offsets[2],
        clip_boxes=clip_boxes,
        variances=variances,
        coords=coords,
        normalize_coords=normalize_coords,
        name="conv6_2_mbox_priorbox",
    )(conv6_2_mbox_loc)
    conv7_2_mbox_priorbox = AnchorBoxes(
        img_height,
        img_width,
        this_scale=scales[3],
        next_scale=scales[4],
        aspect_ratios=aspect_ratios[3],
        two_boxes_for_ar1=two_boxes_for_ar1,
        this_steps=steps[3],
        this_offsets=offsets[3],
        clip_boxes=clip_boxes,
        variances=variances,
        coords=coords,
        normalize_coords=normalize_coords,
        name="conv7_2_mbox_priorbox",
    )(conv7_2_mbox_loc)
    conv8_2_mbox_priorbox = AnchorBoxes(
        img_height,
        img_width,
        this_scale=scales[4],
        next_scale=scales[5],
        aspect_ratios=aspect_ratios[4],
        two_boxes_for_ar1=two_boxes_for_ar1,
        this_steps=steps[4],
        this_offsets=offsets[4],
        clip_boxes=clip_boxes,
        variances=variances,
        coords=coords,
        normalize_coords=normalize_coords,
        name="conv8_2_mbox_priorbox",
    )(conv8_2_mbox_loc)
    conv9_2_mbox_priorbox = AnchorBoxes(
        img_height,
        img_width,
        this_scale=scales[5],
        next_scale=scales[6],
        aspect_ratios=aspect_ratios[5],
        two_boxes_for_ar1=two_boxes_for_ar1,
        this_steps=steps[5],
        this_offsets=offsets[5],
        clip_boxes=clip_boxes,
        variances=variances,
        coords=coords,
        normalize_coords=normalize_coords,
        name="conv9_2_mbox_priorbox",
    )(conv9_2_mbox_loc)

    ### Reshape

    # Reshape the class predictions, yielding 3D tensors of shape `(batch, height * width * n_boxes, n_classes)`
    # We want the classes isolated in the last axis to perform softmax on them
    conv4_3_norm_mbox_conf_reshape = Reshape(
        (-1, n_classes), name="conv4_3_norm_mbox_conf_reshape"
    )(conv4_3_norm_mbox_conf)
    fc7_mbox_conf_reshape = Reshape((-1, n_classes), name="fc7_mbox_conf_reshape")(
        fc7_mbox_conf
    )
    conv6_2_mbox_conf_reshape = Reshape(
        (-1, n_classes), name="conv6_2_mbox_conf_reshape"
    )(conv6_2_mbox_conf)
    conv7_2_mbox_conf_reshape = Reshape(
        (-1, n_classes), name="conv7_2_mbox_conf_reshape"
    )(conv7_2_mbox_conf)
    conv8_2_mbox_conf_reshape = Reshape(
        (-1, n_classes), name="conv8_2_mbox_conf_reshape"
    )(conv8_2_mbox_conf)
    conv9_2_mbox_conf_reshape = Reshape(
        (-1, n_classes), name="conv9_2_mbox_conf_reshape"
    )(conv9_2_mbox_conf)
    # Reshape the box predictions, yielding 3D tensors of shape `(batch, height * width * n_boxes, 4)`
    # We want the four box coordinates isolated in the last axis to compute the smooth L1 loss
    conv4_3_norm_mbox_loc_reshape = Reshape(
        (-1, 4), name="conv4_3_norm_mbox_loc_reshape"
    )(conv4_3_norm_mbox_loc)
    fc7_mbox_loc_reshape = Reshape((-1, 4), name="fc7_mbox_loc_reshape")(fc7_mbox_loc)
    conv6_2_mbox_loc_reshape = Reshape((-1, 4), name="conv6_2_mbox_loc_reshape")(
        conv6_2_mbox_loc
    )
    conv7_2_mbox_loc_reshape = Reshape((-1, 4), name="conv7_2_mbox_loc_reshape")(
        conv7_2_mbox_loc
    )
    conv8_2_mbox_loc_reshape = Reshape((-1, 4), name="conv8_2_mbox_loc_reshape")(
        conv8_2_mbox_loc
    )
    conv9_2_mbox_loc_reshape = Reshape((-1, 4), name="conv9_2_mbox_loc_reshape")(
        conv9_2_mbox_loc
    )
    # Reshape the anchor box tensors, yielding 3D tensors of shape `(batch, height * width * n_boxes, 8)`
    conv4_3_norm_mbox_priorbox_reshape = Reshape(
        (-1, 8), name="conv4_3_norm_mbox_priorbox_reshape"
    )(conv4_3_norm_mbox_priorbox)
    fc7_mbox_priorbox_reshape = Reshape((-1, 8), name="fc7_mbox_priorbox_reshape")(
        fc7_mbox_priorbox
    )
    conv6_2_mbox_priorbox_reshape = Reshape(
        (-1, 8), name="conv6_2_mbox_priorbox_reshape"
    )(conv6_2_mbox_priorbox)
    conv7_2_mbox_priorbox_reshape = Reshape(
        (-1, 8), name="conv7_2_mbox_priorbox_reshape"
    )(conv7_2_mbox_priorbox)
    conv8_2_mbox_priorbox_reshape = Reshape(
        (-1, 8), name="conv8_2_mbox_priorbox_reshape"
    )(conv8_2_mbox_priorbox)
    conv9_2_mbox_priorbox_reshape = Reshape(
        (-1, 8), name="conv9_2_mbox_priorbox_reshape"
    )(conv9_2_mbox_priorbox)

    ### Concatenate the predictions from the different layers

    # Axis 0 (batch) and axis 2 (n_classes or 4, respectively) are identical for all layer predictions,
    # so we want to concatenate along axis 1, the number of boxes per layer
    # Output shape of `mbox_conf`: (batch, n_boxes_total, n_classes)
    mbox_conf = Concatenate(axis=1, name="mbox_conf")(
        [
            conv4_3_norm_mbox_conf_reshape,
            fc7_mbox_conf_reshape,
            conv6_2_mbox_conf_reshape,
            conv7_2_mbox_conf_reshape,
            conv8_2_mbox_conf_reshape,
            conv9_2_mbox_conf_reshape,
        ]
    )

    # Output shape of `mbox_loc`: (batch, n_boxes_total, 4)
    mbox_loc = Concatenate(axis=1, name="mbox_loc")(
        [
            conv4_3_norm_mbox_loc_reshape,
            fc7_mbox_loc_reshape,
            conv6_2_mbox_loc_reshape,
            conv7_2_mbox_loc_reshape,
            conv8_2_mbox_loc_reshape,
            conv9_2_mbox_loc_reshape,
        ]
    )

    # Output shape of `mbox_priorbox`: (batch, n_boxes_total, 8)
    mbox_priorbox = Concatenate(axis=1, name="mbox_priorbox")(
        [
            conv4_3_norm_mbox_priorbox_reshape,
            fc7_mbox_priorbox_reshape,
            conv6_2_mbox_priorbox_reshape,
            conv7_2_mbox_priorbox_reshape,
            conv8_2_mbox_priorbox_reshape,
            conv9_2_mbox_priorbox_reshape,
        ]
    )

    # The box coordinate predictions will go into the loss function just the way they are,
    # but for the class predictions, we'll apply a softmax activation layer first
    mbox_conf_softmax = Activation("softmax", name="mbox_conf_softmax")(mbox_conf)

    # Concatenate the class and box predictions and the anchors to one large predictions vector
    # Output shape of `predictions`: (batch, n_boxes_total, n_classes + 4 + 8)
    predictions = Concatenate(axis=2, name="predictions_ssd")(
        [mbox_conf_softmax, mbox_loc, mbox_priorbox]
    )

    if mode == "training":
        model = Model(inputs=[input_y, input_cbcr], outputs=predictions)

    elif mode == "inference":
        decoded_predictions = DecodeDetections(
            confidence_thresh=confidence_thresh,
            iou_threshold=iou_threshold,
            top_k=top_k,
            nms_max_output_size=nms_max_output_size,
            coords=coords,
            normalize_coords=normalize_coords,
            img_height=img_height,
            img_width=img_width,
            name="decoded_predictions",
        )(predictions)
        model = Model(inputs=[input_y, input_cbcr], outputs=decoded_predictions)

    elif mode == "inference_fast":
        decoded_predictions = DecodeDetectionsFast(
            confidence_thresh=confidence_thresh,
            iou_threshold=iou_threshold,
            top_k=top_k,
            nms_max_output_size=nms_max_output_size,
            coords=coords,
            normalize_coords=normalize_coords,
            img_height=img_height,
            img_width=img_width,
            name="decoded_predictions",
        )(predictions)
        model = Model(inputs=[input_y, input_cbcr], outputs=decoded_predictions)

    else:
        raise ValueError(
            "`mode` must be one of 'training', 'inference' or 'inference_fast', but received '{}'.".format(
                mode
            )
        )

    if return_predictor_sizes:
        predictor_sizes = np.array(
            [
                conv4_3_norm_mbox_conf._keras_shape[1:3],
                fc7_mbox_conf._keras_shape[1:3],
                conv6_2_mbox_conf._keras_shape[1:3],
                conv7_2_mbox_conf._keras_shape[1:3],
                conv8_2_mbox_conf._keras_shape[1:3],
                conv9_2_mbox_conf._keras_shape[1:3],
            ]
        )
        return model, predictor_sizes
    else:
        return model


def ssd_resnet_EF_layers_identical(
    image_size,
    n_classes,
    mode="training",
    l2_regularization=0.0005,
    min_scale=None,
    max_scale=None,
    scales=None,
    aspect_ratios_global=None,
    aspect_ratios_per_layer=[
        [1.0, 2.0, 0.5],
        [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
        [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
        [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
        [1.0, 2.0, 0.5],
        [1.0, 2.0, 0.5],
    ],
    two_boxes_for_ar1=True,
    steps=[8, 16, 32, 64, 100, 300],
    offsets=None,
    clip_boxes=False,
    variances=[0.1, 0.1, 0.2, 0.2],
    coords="centroids",
    normalize_coords=True,
    subtract_mean=[123, 117, 104],
    divide_by_stddev=None,
    swap_channels=[2, 1, 0],
    confidence_thresh=0.01,
    iou_threshold=0.45,
    top_k=200,
    nms_max_output_size=400,
    return_predictor_sizes=False,
    archi="deconv"
):

    n_predictor_layers = 6  # The number of predictor conv layers in the network is 6 for the original SSD300.
    n_classes += 1  # Account for the background class.
    l2_reg = l2_regularization  # Make the internal name shorter.
    img_height, img_width, img_channels = image_size[0], image_size[1], image_size[2]

    ############################################################################
    # Get a few exceptions out of the way.
    ############################################################################

    if aspect_ratios_global is None and aspect_ratios_per_layer is None:
        raise ValueError(
            "`aspect_ratios_global` and `aspect_ratios_per_layer` cannot both be None. At least one needs to be specified."
        )
    if aspect_ratios_per_layer:
        if len(aspect_ratios_per_layer) != n_predictor_layers:
            raise ValueError(
                "It must be either aspect_ratios_per_layer is None or len(aspect_ratios_per_layer) == {}, but len(aspect_ratios_per_layer) == {}.".format(
                    n_predictor_layers, len(aspect_ratios_per_layer)
                )
            )

    if (min_scale is None or max_scale is None) and scales is None:
        raise ValueError(
            "Either `min_scale` and `max_scale` or `scales` need to be specified."
        )
    if scales:
        if len(scales) != n_predictor_layers + 1:
            raise ValueError(
                "It must be either scales is None or len(scales) == {}, but len(scales) == {}.".format(
                    n_predictor_layers + 1, len(scales)
                )
            )
    else:  # If no explicit list of scaling factors was passed, compute the list of scaling factors from `min_scale` and `max_scale`
        scales = np.linspace(min_scale, max_scale, n_predictor_layers + 1)

    if len(variances) != 4:
        raise ValueError(
            "4 variance values must be pased, but {} values were received.".format(
                len(variances)
            )
        )
    variances = np.array(variances)
    if np.any(variances <= 0):
        raise ValueError(
            "All variances must be >0, but the variances given are {}".format(variances)
        )

    if (not (steps is None)) and (len(steps) != n_predictor_layers):
        raise ValueError(
            "You must provide at least one step value per predictor layer."
        )

    if (not (offsets is None)) and (len(offsets) != n_predictor_layers):
        raise ValueError(
            "You must provide at least one offset value per predictor layer."
        )

    ############################################################################
    # Compute the anchor box parameters.
    ############################################################################

    # Set the aspect ratios for each predictor layer. These are only needed for the anchor box layers.
    if aspect_ratios_per_layer:
        aspect_ratios = aspect_ratios_per_layer
    else:
        aspect_ratios = [aspect_ratios_global] * n_predictor_layers

    # Compute the number of boxes to be predicted per cell for each predictor layer.
    # We need this so that we know how many channels the predictor layers need to have.
    if aspect_ratios_per_layer:
        n_boxes = []
        for ar in aspect_ratios_per_layer:
            if (1 in ar) & two_boxes_for_ar1:
                n_boxes.append(len(ar) + 1)  # +1 for the second box for aspect ratio 1
            else:
                n_boxes.append(len(ar))
    else:  # If only a global aspect ratio list was passed, then the number of boxes is the same for each predictor layer
        if (1 in aspect_ratios_global) & two_boxes_for_ar1:
            n_boxes = len(aspect_ratios_global) + 1
        else:
            n_boxes = len(aspect_ratios_global)
        n_boxes = [n_boxes] * n_predictor_layers

    if steps is None:
        steps = [None] * n_predictor_layers
    if offsets is None:
        offsets = [None] * n_predictor_layers

    ############################################################################
    # Define functions for the Lambda layers below.
    ############################################################################

    def identity_layer(tensor):
        return tensor

    def input_mean_normalization(tensor):
        return tensor - np.array(subtract_mean)

    def input_stddev_normalization(tensor):
        return tensor / np.array(divide_by_stddev)

    def input_channel_swap(tensor):
        if len(swap_channels) == 3:
            return K.stack(
                [
                    tensor[..., swap_channels[0]],
                    tensor[..., swap_channels[1]],
                    tensor[..., swap_channels[2]],
                ],
                axis=-1,
            )
        elif len(swap_channels) == 4:
            return K.stack(
                [
                    tensor[..., swap_channels[0]],
                    tensor[..., swap_channels[1]],
                    tensor[..., swap_channels[2]],
                    tensor[..., swap_channels[3]],
                ],
                axis=-1,
            )

    ############################################################################
    # Build the network.
    ############################################################################
    
    if archi == "deconv":
        input_cbcr = None
        x, input_shape, input_y, input_cb, input_cr = deconv()
    else:
        input_cr = None
        if archi == "y_cb4_cbcr_cb5":
            x, input_shape, input_y, input_cbcr = y_in_CB4_cbcr_in_cb5()
        elif archi == "up_sampling":
            x, input_shape, input_y, input_cbcr = up_sampling_rfa()
        elif archi == "cb5_only":
            x, input_shape, input_y, input_cbcr = only_cb5()
        else:
            raise ValueError("Unknown network architecture")

    pool5 = MaxPooling2D((3, 3), strides=(1, 1), padding="same", name="pool5_ssd")(x)

    fc6 = Conv2D(
        1024,
        (3, 3),
        dilation_rate=(6, 6),
        activation="relu",
        padding="same",
        kernel_initializer="he_normal",
        kernel_regularizer=l2(l2_reg),
        name="fc6",
    )(pool5)

    fc7 = Conv2D(
        1024,
        (1, 1),
        activation="relu",
        padding="same",
        kernel_initializer="he_normal",
        kernel_regularizer=l2(l2_reg),
        name="fc7",
    )(fc6)

    conv6_1 = Conv2D(
        256,
        (1, 1),
        activation="relu",
        padding="same",
        kernel_initializer="he_normal",
        kernel_regularizer=l2(l2_reg),
        name="conv6_1",
    )(fc7)

    conv6_1 = ZeroPadding2D(padding=((1, 1), (1, 1)), name="conv6_padding")(conv6_1)

    conv6_2 = Conv2D(
        512,
        (3, 3),
        strides=(2, 2),
        activation="relu",
        padding="valid",
        kernel_initializer="he_normal",
        kernel_regularizer=l2(l2_reg),
        name="conv6_2",
    )(conv6_1)

    conv7_1 = Conv2D(
        128,
        (1, 1),
        activation="relu",
        padding="same",
        kernel_initializer="he_normal",
        kernel_regularizer=l2(l2_reg),
        name="conv7_1",
    )(conv6_2)

    conv7_1 = ZeroPadding2D(padding=((1, 1), (1, 1)), name="conv7_padding")(conv7_1)

    conv7_2 = Conv2D(
        256,
        (3, 3),
        strides=(1, 1),
        activation="relu",
        padding="valid",
        kernel_initializer="he_normal",
        kernel_regularizer=l2(l2_reg),
        name="conv7_2",
    )(conv7_1)

    conv8_1 = Conv2D(
        128,
        (1, 1),
        activation="relu",
        padding="same",
        kernel_initializer="he_normal",
        kernel_regularizer=l2(l2_reg),
        name="conv8_1",
    )(conv7_2)

    conv8_2 = Conv2D(
        256,
        (3, 3),
        strides=(1, 1),
        activation="relu",
        padding="valid",
        kernel_initializer="he_normal",
        kernel_regularizer=l2(l2_reg),
        name="conv8_2",
    )(conv8_1)

    conv9_1 = Conv2D(
        128,
        (1, 1),
        activation="relu",
        padding="same",
        kernel_initializer="he_normal",
        kernel_regularizer=l2(l2_reg),
        name="conv9_1",
    )(conv8_2)
    conv9_2 = Conv2D(
        256,
        (3, 3),
        strides=(1, 1),
        activation="relu",
        padding="valid",
        kernel_initializer="he_normal",
        kernel_regularizer=l2(l2_reg),
        name="conv9_2",
    )(conv9_1)

    # Feed conv4_3 into the L2 normalization layer
    conv4_3_norm = L2Normalization(gamma_init=20, name="conv4_3_norm")(input_y)

    ### Build the convolutional predictor layers on top of the base network

    # We precidt `n_classes` confidence values for each box, hence the confidence predictors have depth `n_boxes * n_classes`
    # Output shape of the confidence layers: `(batch, height, width, n_boxes * n_classes)`
    conv4_3_norm_mbox_conf = Conv2D(
        n_boxes[0] * n_classes,
        (3, 3),
        padding="same",
        kernel_initializer="he_normal",
        kernel_regularizer=l2(l2_reg),
        name="conv4_3_norm_mbox_conf_{}".format(n_classes),
    )(conv4_3_norm)
    fc7_mbox_conf = Conv2D(
        n_boxes[1] * n_classes,
        (3, 3),
        padding="same",
        kernel_initializer="he_normal",
        kernel_regularizer=l2(l2_reg),
        name="fc7_mbox_conf_{}".format(n_classes),
    )(fc7)
    conv6_2_mbox_conf = Conv2D(
        n_boxes[2] * n_classes,
        (3, 3),
        padding="same",
        kernel_initializer="he_normal",
        kernel_regularizer=l2(l2_reg),
        name="conv6_2_mbox_conf_{}".format(n_classes),
    )(conv6_2)
    conv7_2_mbox_conf = Conv2D(
        n_boxes[3] * n_classes,
        (3, 3),
        padding="same",
        kernel_initializer="he_normal",
        kernel_regularizer=l2(l2_reg),
        name="conv7_2_mbox_conf_{}".format(n_classes),
    )(conv7_2)
    conv8_2_mbox_conf = Conv2D(
        n_boxes[4] * n_classes,
        (3, 3),
        padding="same",
        kernel_initializer="he_normal",
        kernel_regularizer=l2(l2_reg),
        name="conv8_2_mbox_conf_{}".format(n_classes),
    )(conv8_2)
    conv9_2_mbox_conf = Conv2D(
        n_boxes[5] * n_classes,
        (3, 3),
        padding="same",
        kernel_initializer="he_normal",
        kernel_regularizer=l2(l2_reg),
        name="conv9_2_mbox_conf_{}".format(n_classes),
    )(conv9_2)
    # We predict 4 box coordinates for each box, hence the localization predictors have depth `n_boxes * 4`
    # Output shape of the localization layers: `(batch, height, width, n_boxes * 4)`
    conv4_3_norm_mbox_loc = Conv2D(
        n_boxes[0] * 4,
        (3, 3),
        padding="same",
        kernel_initializer="he_normal",
        kernel_regularizer=l2(l2_reg),
        name="conv4_3_norm_mbox_loc",
    )(conv4_3_norm)
    fc7_mbox_loc = Conv2D(
        n_boxes[1] * 4,
        (3, 3),
        padding="same",
        kernel_initializer="he_normal",
        kernel_regularizer=l2(l2_reg),
        name="fc7_mbox_loc",
    )(fc7)
    conv6_2_mbox_loc = Conv2D(
        n_boxes[2] * 4,
        (3, 3),
        padding="same",
        kernel_initializer="he_normal",
        kernel_regularizer=l2(l2_reg),
        name="conv6_2_mbox_loc",
    )(conv6_2)
    conv7_2_mbox_loc = Conv2D(
        n_boxes[3] * 4,
        (3, 3),
        padding="same",
        kernel_initializer="he_normal",
        kernel_regularizer=l2(l2_reg),
        name="conv7_2_mbox_loc",
    )(conv7_2)
    conv8_2_mbox_loc = Conv2D(
        n_boxes[4] * 4,
        (3, 3),
        padding="same",
        kernel_initializer="he_normal",
        kernel_regularizer=l2(l2_reg),
        name="conv8_2_mbox_loc",
    )(conv8_2)
    conv9_2_mbox_loc = Conv2D(
        n_boxes[5] * 4,
        (3, 3),
        padding="same",
        kernel_initializer="he_normal",
        kernel_regularizer=l2(l2_reg),
        name="conv9_2_mbox_loc",
    )(conv9_2)

    ### Generate the anchor boxes (called "priors" in the original Caffe/C++ implementation, so I'll keep their layer names)

    # Output shape of anchors: `(batch, height, width, n_boxes, 8)`
    conv4_3_norm_mbox_priorbox = AnchorBoxes(
        img_height,
        img_width,
        this_scale=scales[0],
        next_scale=scales[1],
        aspect_ratios=aspect_ratios[0],
        two_boxes_for_ar1=two_boxes_for_ar1,
        this_steps=steps[0],
        this_offsets=offsets[0],
        clip_boxes=clip_boxes,
        variances=variances,
        coords=coords,
        normalize_coords=normalize_coords,
        name="conv4_3_norm_mbox_priorbox",
    )(conv4_3_norm_mbox_loc)
    fc7_mbox_priorbox = AnchorBoxes(
        img_height,
        img_width,
        this_scale=scales[1],
        next_scale=scales[2],
        aspect_ratios=aspect_ratios[1],
        two_boxes_for_ar1=two_boxes_for_ar1,
        this_steps=steps[1],
        this_offsets=offsets[1],
        clip_boxes=clip_boxes,
        variances=variances,
        coords=coords,
        normalize_coords=normalize_coords,
        name="fc7_mbox_priorbox",
    )(fc7_mbox_loc)
    conv6_2_mbox_priorbox = AnchorBoxes(
        img_height,
        img_width,
        this_scale=scales[2],
        next_scale=scales[3],
        aspect_ratios=aspect_ratios[2],
        two_boxes_for_ar1=two_boxes_for_ar1,
        this_steps=steps[2],
        this_offsets=offsets[2],
        clip_boxes=clip_boxes,
        variances=variances,
        coords=coords,
        normalize_coords=normalize_coords,
        name="conv6_2_mbox_priorbox",
    )(conv6_2_mbox_loc)
    conv7_2_mbox_priorbox = AnchorBoxes(
        img_height,
        img_width,
        this_scale=scales[3],
        next_scale=scales[4],
        aspect_ratios=aspect_ratios[3],
        two_boxes_for_ar1=two_boxes_for_ar1,
        this_steps=steps[3],
        this_offsets=offsets[3],
        clip_boxes=clip_boxes,
        variances=variances,
        coords=coords,
        normalize_coords=normalize_coords,
        name="conv7_2_mbox_priorbox",
    )(conv7_2_mbox_loc)
    conv8_2_mbox_priorbox = AnchorBoxes(
        img_height,
        img_width,
        this_scale=scales[4],
        next_scale=scales[5],
        aspect_ratios=aspect_ratios[4],
        two_boxes_for_ar1=two_boxes_for_ar1,
        this_steps=steps[4],
        this_offsets=offsets[4],
        clip_boxes=clip_boxes,
        variances=variances,
        coords=coords,
        normalize_coords=normalize_coords,
        name="conv8_2_mbox_priorbox",
    )(conv8_2_mbox_loc)
    conv9_2_mbox_priorbox = AnchorBoxes(
        img_height,
        img_width,
        this_scale=scales[5],
        next_scale=scales[6],
        aspect_ratios=aspect_ratios[5],
        two_boxes_for_ar1=two_boxes_for_ar1,
        this_steps=steps[5],
        this_offsets=offsets[5],
        clip_boxes=clip_boxes,
        variances=variances,
        coords=coords,
        normalize_coords=normalize_coords,
        name="conv9_2_mbox_priorbox",
    )(conv9_2_mbox_loc)

    ### Reshape

    # Reshape the class predictions, yielding 3D tensors of shape `(batch, height * width * n_boxes, n_classes)`
    # We want the classes isolated in the last axis to perform softmax on them
    conv4_3_norm_mbox_conf_reshape = Reshape(
        (-1, n_classes), name="conv4_3_norm_mbox_conf_reshape"
    )(conv4_3_norm_mbox_conf)
    fc7_mbox_conf_reshape = Reshape((-1, n_classes), name="fc7_mbox_conf_reshape")(
        fc7_mbox_conf
    )
    conv6_2_mbox_conf_reshape = Reshape(
        (-1, n_classes), name="conv6_2_mbox_conf_reshape"
    )(conv6_2_mbox_conf)
    conv7_2_mbox_conf_reshape = Reshape(
        (-1, n_classes), name="conv7_2_mbox_conf_reshape"
    )(conv7_2_mbox_conf)
    conv8_2_mbox_conf_reshape = Reshape(
        (-1, n_classes), name="conv8_2_mbox_conf_reshape"
    )(conv8_2_mbox_conf)
    conv9_2_mbox_conf_reshape = Reshape(
        (-1, n_classes), name="conv9_2_mbox_conf_reshape"
    )(conv9_2_mbox_conf)
    # Reshape the box predictions, yielding 3D tensors of shape `(batch, height * width * n_boxes, 4)`
    # We want the four box coordinates isolated in the last axis to compute the smooth L1 loss
    conv4_3_norm_mbox_loc_reshape = Reshape(
        (-1, 4), name="conv4_3_norm_mbox_loc_reshape"
    )(conv4_3_norm_mbox_loc)
    fc7_mbox_loc_reshape = Reshape((-1, 4), name="fc7_mbox_loc_reshape")(fc7_mbox_loc)
    conv6_2_mbox_loc_reshape = Reshape((-1, 4), name="conv6_2_mbox_loc_reshape")(
        conv6_2_mbox_loc
    )
    conv7_2_mbox_loc_reshape = Reshape((-1, 4), name="conv7_2_mbox_loc_reshape")(
        conv7_2_mbox_loc
    )
    conv8_2_mbox_loc_reshape = Reshape((-1, 4), name="conv8_2_mbox_loc_reshape")(
        conv8_2_mbox_loc
    )
    conv9_2_mbox_loc_reshape = Reshape((-1, 4), name="conv9_2_mbox_loc_reshape")(
        conv9_2_mbox_loc
    )
    # Reshape the anchor box tensors, yielding 3D tensors of shape `(batch, height * width * n_boxes, 8)`
    conv4_3_norm_mbox_priorbox_reshape = Reshape(
        (-1, 8), name="conv4_3_norm_mbox_priorbox_reshape"
    )(conv4_3_norm_mbox_priorbox)
    fc7_mbox_priorbox_reshape = Reshape((-1, 8), name="fc7_mbox_priorbox_reshape")(
        fc7_mbox_priorbox
    )
    conv6_2_mbox_priorbox_reshape = Reshape(
        (-1, 8), name="conv6_2_mbox_priorbox_reshape"
    )(conv6_2_mbox_priorbox)
    conv7_2_mbox_priorbox_reshape = Reshape(
        (-1, 8), name="conv7_2_mbox_priorbox_reshape"
    )(conv7_2_mbox_priorbox)
    conv8_2_mbox_priorbox_reshape = Reshape(
        (-1, 8), name="conv8_2_mbox_priorbox_reshape"
    )(conv8_2_mbox_priorbox)
    conv9_2_mbox_priorbox_reshape = Reshape(
        (-1, 8), name="conv9_2_mbox_priorbox_reshape"
    )(conv9_2_mbox_priorbox)

    ### Concatenate the predictions from the different layers

    # Axis 0 (batch) and axis 2 (n_classes or 4, respectively) are identical for all layer predictions,
    # so we want to concatenate along axis 1, the number of boxes per layer
    # Output shape of `mbox_conf`: (batch, n_boxes_total, n_classes)
    mbox_conf = Concatenate(axis=1, name="mbox_conf")(
        [
            conv4_3_norm_mbox_conf_reshape,
            fc7_mbox_conf_reshape,
            conv6_2_mbox_conf_reshape,
            conv7_2_mbox_conf_reshape,
            conv8_2_mbox_conf_reshape,
            conv9_2_mbox_conf_reshape,
        ]
    )

    # Output shape of `mbox_loc`: (batch, n_boxes_total, 4)
    mbox_loc = Concatenate(axis=1, name="mbox_loc")(
        [
            conv4_3_norm_mbox_loc_reshape,
            fc7_mbox_loc_reshape,
            conv6_2_mbox_loc_reshape,
            conv7_2_mbox_loc_reshape,
            conv8_2_mbox_loc_reshape,
            conv9_2_mbox_loc_reshape,
        ]
    )

    # Output shape of `mbox_priorbox`: (batch, n_boxes_total, 8)
    mbox_priorbox = Concatenate(axis=1, name="mbox_priorbox")(
        [
            conv4_3_norm_mbox_priorbox_reshape,
            fc7_mbox_priorbox_reshape,
            conv6_2_mbox_priorbox_reshape,
            conv7_2_mbox_priorbox_reshape,
            conv8_2_mbox_priorbox_reshape,
            conv9_2_mbox_priorbox_reshape,
        ]
    )

    # The box coordinate predictions will go into the loss function just the way they are,
    # but for the class predictions, we'll apply a softmax activation layer first
    mbox_conf_softmax = Activation("softmax", name="mbox_conf_softmax")(mbox_conf)

    # Concatenate the class and box predictions and the anchors to one large predictions vector
    # Output shape of `predictions`: (batch, n_boxes_total, n_classes + 4 + 8)
    predictions = Concatenate(axis=2, name="predictions_ssd")(
        [mbox_conf_softmax, mbox_loc, mbox_priorbox]
    )

    if mode == "training":
        if input_cr is None:
            model = Model(inputs=[input_y, input_cbcr], outputs=predictions)
        else:
            model = Model(inputs=[input_y, input_cb, input_cr], outputs=predictions)
        
    elif mode == "inference":
        decoded_predictions = DecodeDetections(
            confidence_thresh=confidence_thresh,
            iou_threshold=iou_threshold,
            top_k=top_k,
            nms_max_output_size=nms_max_output_size,
            coords=coords,
            normalize_coords=normalize_coords,
            img_height=img_height,
            img_width=img_width,
            name="decoded_predictions",
        )(predictions)
        if input_cr is None:
            model = Model(inputs=[input_y, input_cbcr], outputs=decoded_predictions)
        else:
            model = Model(inputs=[input_y, input_cb, input_cr], outputs=decoded_predictions)
    elif mode == "inference_fast":
        decoded_predictions = DecodeDetectionsFast(
            confidence_thresh=confidence_thresh,
            iou_threshold=iou_threshold,
            top_k=top_k,
            nms_max_output_size=nms_max_output_size,
            coords=coords,
            normalize_coords=normalize_coords,
            img_height=img_height,
            img_width=img_width,
            name="decoded_predictions",
        )(predictions)
        if input_cr is None:
            model = Model(inputs=[input_y, input_cbcr], outputs=decoded_predictions)
        else:
            model = Model(inputs=[input_y, input_cb, input_cr], outputs=decoded_predictions)
    else:
        raise ValueError(
            "`mode` must be one of 'training', 'inference' or 'inference_fast', but received '{}'.".format(
                mode
            )
        )

    if return_predictor_sizes:
        predictor_sizes = np.array(
            [
                conv4_3_norm_mbox_conf._keras_shape[1:3],
                fc7_mbox_conf._keras_shape[1:3],
                conv6_2_mbox_conf._keras_shape[1:3],
                conv7_2_mbox_conf._keras_shape[1:3],
                conv8_2_mbox_conf._keras_shape[1:3],
                conv9_2_mbox_conf._keras_shape[1:3],
            ]
        )
        return model, predictor_sizes
    else:
        return model


def y_in_CB4_cbcr_in_cb5():
    input_shape_y = (38, 38, 64)
    input_shape_cbcr = (19, 19, 128)

    input_y = Input(input_shape_y)
    input_cbcr = Input(input_shape_cbcr)

    y = BatchNormalization(input_shape=input_shape_y)(input_y)
    y = conv_block(y, 1, [256, 256, 384], stage=1, block="a2", strides=(1, 1))
    y = identity_block(y, 2, [256, 256, 384], stage=1, block="b2")
    y = identity_block(y, 3, [256, 256, 384], stage=1, block="c2")

    y = conv_block(y, 3, [128, 128, 512], stage=2, block="a3", strides=(1, 1))
    y = identity_block(y, 3, [128, 128, 512], stage=2, block="b3")
    y = identity_block(y, 3, [128, 128, 512], stage=2, block="c3")
    conv4_3 = identity_block(y, 3, [128, 128, 512], stage=2, block="d3")

    y = conv_block(conv4_3, 3, [256, 256, 384], stage=2, block='a4', strides=(1, 1))

    # Block 4
    x = conv_block(conv4_3, 3, [256, 256, 768], stage=4, block="a2")
    x = identity_block(x, 3, [256, 256, 768], stage=4, block="b2")
    x = identity_block(x, 3, [256, 256, 768], stage=4, block="c2")
    x = identity_block(x, 3, [256, 256, 768], stage=4, block="d2")
    x = identity_block(x, 3, [256, 256, 768], stage=4, block="e2")
    conv4_6 = identity_block(x, 3, [256, 256, 768], stage=4, block="f2")

    cbcr = BatchNormalization(input_shape=input_shape_cbcr)(input_cbcr)
    cbcr = conv_block(cbcr, 1, [256, 256, 256], stage=2, block="a5", strides=(1, 1))
    x = Concatenate(axis=-1)([conv4_6, cbcr])

    # Block 5
    x = conv_block(x, 3, [512, 512, 2048], stage=5, block="a")
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block="b")
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block="c")

    return x, [38, 38, input_shape_y[2] + input_shape_cbcr[2]], input_y, input_cbcr

def up_sampling():
    input_shape_y = (38, 38, 64)
    input_shape_cbcr = (19, 19, 128)

    input_y = Input(input_shape_y)
    input_cbcr = Input(input_shape_cbcr)

    cbcr = UpSampling2D()(input_cbcr)

    concat = Concatenate(axis=-1)([input_y, cbcr])
    x = BatchNormalization(input_shape=input_shape_y)(concat)

    # Block 3
    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a1', strides=(1, 1))
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')

    # Block 4
    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f')

    # Block 5
    x = conv_block(x, 3, [512, 512, 2048], stage=5, block="a")
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block="b")
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block="c")

    return x, [38, 38, input_shape_y[2] + input_shape_cbcr[2]], input_y, input_cbcr

def up_sampling_rfa():
    input_shape_y = (38, 38, 64)
    input_shape_cbcr = (19, 19, 128)

    input_y = Input(input_shape_y)
    input_cbcr = Input(input_shape_cbcr)

    cbcr = UpSampling2D()(input_cbcr)

    concat = Concatenate(axis=-1)([input_y, cbcr])

    x = BatchNormalization(input_shape=input_shape_y)(concat)

    x = conv_block(x, 1, [256, 256, 1024], stage=4, block="a2", strides=(1, 1))
    x = identity_block(x, 2, [256, 256, 1024], stage=4, block="b2")
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block="c2")
    
    # Block 3
    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a1', strides=(1, 1))
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')

    # Block 4
    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
    conv4_3 = identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
    x = identity_block(conv4_3, 3, [256, 256, 1024], stage=4, block='d')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f')

    # Block 5
    x = conv_block(x, 3, [512, 512, 2048], stage=5, block="a")
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block="b")
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block="c")

    return x, [38, 38, input_shape_y[2] + input_shape_cbcr[2]], input_y, input_cbcr

def deconv():
    input_shape_y = (38, 38, 64)
    input_shape_cb = (19, 19, 64)
    input_shape_cr = (19, 19, 64)

    input_y = Input(input_shape_y)
    input_cb = Input(input_shape_cb)
    input_cr = Input(input_shape_cr)

    cb = Conv2DTranspose(64, 2, strides=(2,2))(input_cb)

    cr = Conv2DTranspose(64, 2, strides=(2,2))(input_cr)
    
    cbcr = Concatenate(axis=-1)([cb, cr])
    concat = Concatenate(axis=-1)([input_y, cbcr])

    x = BatchNormalization(input_shape=input_shape_y)(concat)

    x = conv_block(x, 1, [256, 256, 1024], stage=4, block="a2", strides=(1, 1))
    x = identity_block(x, 2, [256, 256, 1024], stage=4, block="b2")
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block="c2")
    
    # Block 3
    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a1', strides=(1, 1))
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')

    # Block 4
    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
    conv4_3 = identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
    x = identity_block(conv4_3, 3, [256, 256, 1024], stage=4, block='d')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f')

    # Block 5
    x = conv_block(x, 3, [512, 512, 2048], stage=5, block="a")
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block="b")
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block="c")

    return x, [38, 38, input_shape_y[2] + input_shape_cb[2] + input_shape_cr[2]], input_y, input_cb, input_cr

def only_cb5():
    input_shape_y = (38, 38, 64)
    input_shape_cbcr = (19, 19, 128)

    input_y = Input(input_shape_y)
    input_cbcr = Input(input_shape_cbcr)

    y = BatchNormalization(input_shape=input_shape_y)(input_y)
    y = conv_block(y, 1, [256, 256, 768], stage=1, block="a2", strides=(1, 1))
    y = identity_block(y, 2, [256, 256, 768], stage=1, block="b2")
    y = identity_block(y, 3, [256, 256, 768], stage=1, block="c2")

    y = conv_block(y, 3, [256, 256, 768], stage=2, block="a3", strides=(1, 1))
    y = identity_block(y, 3, [256, 256, 768], stage=2, block="b3")
    y = identity_block(y, 3, [256, 256, 768], stage=2, block="c3")
    conv4_3 = identity_block(y, 3, [256, 256, 768], stage=2, block="d3")
   
    y = conv_block(conv4_3, 3, [256, 256, 768], stage=2, block='a4')

    cbcr = BatchNormalization(input_shape=input_shape_cbcr)(input_cbcr)
    cbcr = conv_block(cbcr, 1, [256, 256, 256], stage=2, block="a5", strides=(1, 1))
    x = Concatenate(axis=-1)([y, cbcr])

    # Block 5
    x = conv_block(x, 3, [512, 512, 2048], stage=5, block="a")
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block="b")
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block="c")

    return x, [38, 38, input_shape_y[2] + input_shape_cbcr[2]], input_y, input_cbcr

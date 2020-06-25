"""ResNet50 model for Keras.

# Reference:

- [Deep Residual Learning for Image Recognition](
    https://arxiv.org/abs/1512.03385) (CVPR 2016 Best Paper Award)

Adapted from code contributed by BigMoyan.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import warnings

# from . import get_submodules_from_kwargs
# from . import imagenet_utils
# from .imagenet_utils import decode_predictions
# from .imagenet_utils import _obtain_input_shape

from keras.layers.normalization import BatchNormalization
from keras.models import Sequential, Model
from keras.layers import (
    Input,
    Activation,
    Dense,
    Flatten,
    Concatenate,
    Add,
    GlobalAveragePooling2D,
    GlobalMaxPooling2D,
    ZeroPadding2D,
    Conv2DTranspose
)

from keras.layers.convolutional import (
    Conv2D,
    MaxPooling2D,
    AveragePooling2D,
    UpSampling2D
)

# preprocess_input = imagenet_utils.preprocess_input

WEIGHTS_PATH = ('https://github.com/fchollet/deep-learning-models/'
                'releases/download/v0.2/'
                'resnet50_weights_tf_dim_ordering_tf_kernels.h5')
WEIGHTS_PATH_NO_TOP = ('https://github.com/fchollet/deep-learning-models/'
                       'releases/download/v0.2/'
                       'resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5')

backend = None
layers = None
models = None
import keras.utils # keras_utils = None


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
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    # res4f_branch2c
    # bn4f_branch2c    

    # filters: Integer, the dimensionality of the output space (i.e. the number of output filters in the convolution).
    # kernel_size: An integer or tuple/list of 2 integers, specifying the height and width of the 2D convolution window. 
    # Can be a single integer to specify the same value for all spatial dimensions.

    x = Conv2D(filters1, (1, 1),
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size,
                      padding='same',
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1),
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    x = Add()([x, input_tensor])
    x = Activation('relu')(x)
    return x


def conv_block(input_tensor,
               kernel_size,
               filters,
               stage,
               block,
               strides=(2, 2)):
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

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1), strides=strides,
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size, padding='same',
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1),
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    shortcut = Conv2D(filters3, (1, 1), strides=strides,
                             kernel_initializer='he_normal',
                             name=conv_name_base + '1')(input_tensor)
    shortcut = BatchNormalization(
        axis=bn_axis, name=bn_name_base + '1')(shortcut)

    x = Add()([x, shortcut])
    x = Activation('relu')(x)
    return x

def ResNet50RGB(include_top=True,
             weights='imagenet',
             input_tensor=None,
             input_shape=(224,224,3),
             pooling=None,
             classes=1000,
             **kwargs):
    """Instantiates the ResNet50 architecture.

    Optionally loads weights pre-trained on ImageNet.
    Note that the data format convention used by the model is
    the one specified in your Keras config at `~/.keras/keras.json`.

    # Arguments
        include_top: whether to include the fully-connected
            layer at the top of the network.
        weights: one of `None` (random initialization),
              'imagenet' (pre-training on ImageNet),
              or the path to the weights file to be loaded.
        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
            to use as image input for the model.
        input_shape: optional shape tuple, only to be specified
            if `include_top` is False (otherwise the input shape
            has to be `(224, 224, 3)` (with `channels_last` data format)
            or `(3, 224, 224)` (with `channels_first` data format).
            It should have exactly 3 inputs channels,
            and width and height should be no smaller than 32.
            E.g. `(200, 200, 3)` would be one valid value.
        pooling: Optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model will be
                the 4D tensor output of the
                last convolutional block.
            - `avg` means that global average pooling
                will be applied to the output of the
                last convolutional block, and thus
                the output of the model will be a 2D tensor.
            - `max` means that global max pooling will
                be applied.
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.

    # Returns
        A Keras model instance.

    # Raises
        ValueError: in case of invalid argument for `weights`,
            or invalid input shape.
    """

    global backend, layers, models#, keras_utils
    keras_utils = keras.utils
    # backend, layers, models, keras_utils = get_submodules_from_kwargs(kwargs)
    bn_axis = 3
    if not (weights in {'imagenet', None} or os.path.exists(weights)):
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization), `imagenet` '
                         '(pre-training on ImageNet), '
                         'or the path to the weights file to be loaded.')

    if weights == 'imagenet' and include_top and classes != 1000:
        raise ValueError('If using `weights` as `"imagenet"` with `include_top`'
                         ' as true, `classes` should be 1000')



    # Zero-padding layer for 2D input (e.g. picture).
    # ZeroPadding2D This layer can add rows and columns of zeros at the top, bottom, left and right side of an image tensor.
    
    # padding: int, or tuple of 2 ints, or tuple of 2 tuples of 2 ints.
    # If int: the same symmetric padding is applied to height and width.
    # If tuple of 2 ints: interpreted as two different symmetric padding values for height and width: (symmetric_height_pad, symmetric_width_pad).
    # If tuple of 2 tuples of 2 ints: interpreted as ((top_pad, bottom_pad), (left_pad, right_pad))
    img_input = Input(shape=input_shape)
    x = ZeroPadding2D(padding=(3, 3), name='conv1_pad')(img_input)
    x = Conv2D(64, (7, 7),
                      strides=(2, 2),
                      padding='valid',
                      kernel_initializer='he_normal',
                      name='conv1')(x)
    x = BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
    # 112*112*64
    x = Activation('relu')(x)
    x = ZeroPadding2D(padding=(1, 1), name='pool1_pad')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)
    # 56*56*64

    # Block 2
    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')

    # Block 3
    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
    # 28*28*512
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
    x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')

    if include_top:
        x = GlobalAveragePooling2D(name='avg_pool')(x)
        x = Dense(classes, activation='softmax', name='fc1000')(x)
    else:
        if pooling == 'avg':
            x = GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = GlobalMaxPooling2D()(x)
        else:
            warnings.warn('The output shape of `ResNet50(include_top=False)` '
                          'has been changed since Keras 2.2.0.')

    # Create model.
    inputs = img_input
    model = Model(inputs, x, name='resnet50rgb')

    # Load weights.
    if weights == 'imagenet':
        if include_top:
            weights_path = keras_utils.get_file(
                'resnet50_weights_tf_dim_ordering_tf_kernels.h5',
                WEIGHTS_PATH,
                cache_subdir='models',
                md5_hash='a7b3fe01876f51b976af0dea6bc144eb')
        else:
            weights_path = keras_utils.get_file(
                'resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5',
                WEIGHTS_PATH_NO_TOP,
                cache_subdir='models',
                md5_hash='a268eb855778b3df3c7506639542a6af')
        model.load_weights(weights_path)
        # if backend.backend() == 'theano':
        #     keras_utils.convert_all_kernels_in_model(model)
    elif weights is not None:
        model.load_weights(weights)

    return model


def ResNet50Custom(include_top=True,
             weights='imagenet',
             input_tensor=None,
             input_shape=None,
             pooling=None,
             classes=1000,
             archi="late_concat",
             **kwargs):
    """Instantiates the ResNet50 architecture.

    Optionally loads weights pre-trained on ImageNet.
    Note that the data format convention used by the model is
    the one specified in your Keras config at `~/.keras/keras.json`.

    # Arguments
        include_top: whether to include the fully-connected
            layer at the top of the network.
        weights: one of `None` (random initialization),
              'imagenet' (pre-training on ImageNet),
              or the path to the weights file to be loaded.
        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
            to use as image input for the model.
        input_shape: optional shape tuple, only to be specified
            if `include_top` is False (otherwise the input shape
            has to be `(224, 224, 3)` (with `channels_last` data format)
            or `(3, 224, 224)` (with `channels_first` data format).
            It should have exactly 3 inputs channels,
            and width and height should be no smaller than 32.
            E.g. `(200, 200, 3)` would be one valid value.
        pooling: Optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model will be
                the 4D tensor output of the
                last convolutional block.
            - `avg` means that global average pooling
                will be applied to the output of the
                last convolutional block, and thus
                the output of the model will be a 2D tensor.
            - `max` means that global max pooling will
                be applied.
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.

    # Returns
        A Keras model instance.

    # Raises
        ValueError: in case of invalid argument for `weights`,
            or invalid input shape.
    """

    global backend, layers, models, keras_utils
    # backend, layers, models, keras_utils = get_submodules_from_kwargs(kwargs)
    keras_utils = keras.utils
    if not (weights in {'imagenet', None} or os.path.exists(weights)):
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization), `imagenet` '
                         '(pre-training on ImageNet), '
                         'or the path to the weights file to be loaded.')

    if weights == 'imagenet' and include_top and classes != 1000:
        raise ValueError('If using `weights` as `"imagenet"` with `include_top`'
                         ' as true, `classes` should be 1000')



    # Zero-padding layer for 2D input (e.g. picture).
    # ZeroPadding2D This layer can add rows and columns of zeros at the top, bottom, left and right side of an image tensor.
    
    # padding: int, or tuple of 2 ints, or tuple of 2 tuples of 2 ints.
    # If int: the same symmetric padding is applied to height and width.
    # If tuple of 2 ints: interpreted as two different symmetric padding values for height and width: (symmetric_height_pad, symmetric_width_pad).
    # If tuple of 2 tuples of 2 ints: interpreted as ((top_pad, bottom_pad), (left_pad, right_pad))

    if archi == "deconv":
        x, input_shape, input_y, input_cb, input_cr = deconv()
        input_cbcr = None
    else:
        input_cr = None
        if archi == "late_concat_rfa_thinner":
            x, input_shape, input_y, input_cbcr = late_concat_rfa_thinner()
        elif archi == "up_sampling":
            x, input_shape, input_y, input_cbcr = up_sampling()
        elif archi == "up_sampling_rfa":
            x, input_shape, input_y, input_cbcr = up_sampling_rfa()
        elif archi == "cb5_only":
            x, input_shape, input_y, input_cbcr = only_cb5()
        elif archi == "late_concat_more_channels":
            x, input_shape, input_y, input_cbcr = late_concat_rfa_thinner_more_channels()
        elif archi == "y_cb4_cbcr_cb5":
            x, input_shape, input_y, input_cbcr = y_in_CB4_cbcr_in_cb5()

    # Block 5
    x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')

    if include_top:
        x = GlobalAveragePooling2D(name='avg_pool')(x)
        x = Dense(classes, activation='softmax', name='fc1000')(x)
    else:
        if pooling == 'avg':
            x = GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = GlobalMaxPooling2D()(x)
        else:
            warnings.warn('The output shape of `ResNet50(include_top=False)` '
                          'has been changed since Keras 2.2.0.')

    # Create model.
    if input_cbcr is not None:
        model = Model(inputs=[input_y, input_cbcr], outputs=x, name='resnet50_custom')
    else:
        model = Model(inputs=[input_y, input_cb, input_cr], outputs=x, name='resnet50_custom')
    
    # Load weights.
    if weights == 'imagenet':
        if include_top:
            weights_path = keras_utils.get_file(
                'resnet50_weights_tf_dim_ordering_tf_kernels.h5',
                WEIGHTS_PATH,
                cache_subdir='models',
                md5_hash='a7b3fe01876f51b976af0dea6bc144eb')
        else:
            weights_path = keras_utils.get_file(
                'resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5',
                WEIGHTS_PATH_NO_TOP,
                cache_subdir='models',
                md5_hash='a268eb855778b3df3c7506639542a6af')

        model.load_weights(weights_path, by_name=True)
    elif weights is not None:
        model.load_weights(weights)

    return model

def up_sampling(): 
    # 28*8=224, taille de l'image originale
    # 38*8=304
    input_shape_y = (28, 28, 64)
    input_shape_cbcr = (14, 14, 128)

    input_y = Input(input_shape_y)
    input_cbcr = Input(input_shape_cbcr)

    cbcr = UpSampling2D()(input_cbcr)
    # 28*28

    concat = Concatenate(axis=-1)([input_y, cbcr])
    # 28*28

    x = BatchNormalization(input_shape=input_shape_y)(concat)

    # Block 3
    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a1', strides=(1, 1))
    # 14*14
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')

    # Block 4
    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
    # 7*7
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f')

    return x, [28, 28, input_shape_y[2]+input_shape_cbcr[2]], input_y, input_cbcr

def late_concat_rfa_thinner(): 
    input_shape_y = (28, 28, 64)
    input_shape_cbcr = (14, 14, 128)

    input_y = Input(input_shape_y)
    input_cbcr = Input(input_shape_cbcr)

    y = BatchNormalization(input_shape=input_shape_y)(input_y)
    y = conv_block(y, 1, [256, 256, 384], stage=1, block='a2', strides=(1, 1))
    y = identity_block(y, 2, [256, 256, 384], stage=1, block='b2')
    y = identity_block(y, 3, [256, 256, 384], stage=1, block='c2')

    y = conv_block(y, 3, [128, 128, 384], stage=2, block='a3', strides=(1, 1))
    y = identity_block(y, 3, [128, 128, 384], stage=2, block='b3')
    y = identity_block(y, 3, [128, 128, 384], stage=2, block='c3')
    y = identity_block(y, 3, [128, 128, 384], stage=2, block='d3')

    y = conv_block(y, 3, [256, 256, 384], stage=2, block='a4')

    cbcr = BatchNormalization(input_shape=input_shape_cbcr)(input_cbcr)
    cbcr = conv_block(cbcr, 1, [256, 256, 128], stage=2, block='a5', strides=(1, 1))

    x = Concatenate(axis=-1)([y, cbcr])

    # Block 3
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


    return x, [28, 28, input_shape_y[2]+input_shape_cbcr[2]], input_y, input_cbcr

def late_concat_rfa_thinner_more_channels():
    input_shape_y = (28, 28, 64)
    input_shape_cbcr = (14, 14, 128)

    input_y = Input(input_shape_y)
    input_cbcr = Input(input_shape_cbcr)

    y = BatchNormalization(input_shape=input_shape_y)(input_y)
    y = conv_block(y, 1, [256, 256, 768], stage=1, block='a2', strides=(1, 1))
    y = identity_block(y, 2, [256, 256, 768], stage=1, block='b2')
    y = identity_block(y, 3, [256, 256, 768], stage=1, block='c2')

    y = conv_block(y, 3, [256, 256, 768], stage=2, block='a3', strides=(1, 1))
    y = identity_block(y, 3, [256, 256, 768], stage=2, block='b3')
    y = identity_block(y, 3, [256, 256, 768], stage=2, block='c3')
    y = identity_block(y, 3, [256, 256, 768], stage=2, block='d3')

    y = conv_block(y, 3, [256, 256, 384], stage=2, block='a4')

    cbcr = BatchNormalization(input_shape=input_shape_cbcr)(input_cbcr)
    cbcr = conv_block(cbcr, 1, [256, 256, 128], stage=2, block='a5', strides=(1, 1))

    x = Concatenate(axis=-1)([y, cbcr])

    # Block 3
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b1')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c1')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='d1')

    # Block 4
    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f')

    return x, [28, 28, input_shape_y[2] + input_shape_cbcr[2]], input_y, input_cbcr

def up_sampling_rfa():
    input_shape_y = (28, 28, 64)
    input_shape_cbcr = (14, 14, 128)

    input_y = Input(input_shape_y)
    input_cbcr = Input(input_shape_cbcr)

    cbcr = UpSampling2D()(input_cbcr)
    # 38*38

    concat = Concatenate(axis=-1)([input_y, cbcr])
    # 38*38

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

    return x, [28, 28, input_shape_y[2] + input_shape_cbcr[2]], input_y, input_cbcr

def deconv():
    input_shape_y = (28, 28, 64)
    # input_shape_cbcr = (14, 14, 128)
    input_shape_cb = (14, 14, 64)
    input_shape_cr = (14, 14, 64)

    input_y = Input(input_shape_y)
    # input_cbcr = Input(input_shape_cbcr)
    input_cb = Input(input_shape_cb)
    input_cr = Input(input_shape_cr)

    cb = Conv2DTranspose(64, 2, strides=(2,2))(input_cb)

    cr = Conv2DTranspose(64, 2, strides=(2,2))(input_cr)
    
    cbcr = Concatenate(axis=-1)([cb, cr])
    # cbcr = Conv2DTranspose(128, 2, strides=(2,2))(input_cbcr)
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

    return x, [28, 28, input_shape_y[2] + input_shape_cb[2] + input_shape_cr[2]], input_y, input_cb, input_cr


def only_cb5():
    input_shape_y = (28, 28, 64)
    input_shape_cbcr = (14, 14, 128)

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
    # y : 19*19*384

    cbcr = BatchNormalization(input_shape=input_shape_cbcr)(input_cbcr)
    cbcr = conv_block(cbcr, 1, [256, 256, 256], stage=2, block="a5", strides=(1, 1))
    x = Concatenate(axis=-1)([y, cbcr])

    return x, [28, 28, input_shape_y[2] + input_shape_cbcr[2]], input_y, input_cbcr


def y_in_CB4_cbcr_in_cb5():
    input_shape_y = (28, 28, 64)
    input_shape_cbcr = (14, 14, 128)

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
    # conv4_3 : 38*38*512

    # y = conv_block(conv4_3, 3, [256, 256, 384], stage=2, block='a4')
    y = conv_block(conv4_3, 3, [256, 256, 384], stage=2, block='a4', strides=(1, 1))
    # y : 19*19*384

    
    # cbcr : 19*19*128

    # Block 4
    x = conv_block(conv4_3, 3, [256, 256, 768], stage=4, block="a2")
    # x : 19*19*1024
    x = identity_block(x, 3, [256, 256, 768], stage=4, block="b2")
    x = identity_block(x, 3, [256, 256, 768], stage=4, block="c2")
    x = identity_block(x, 3, [256, 256, 768], stage=4, block="d2")
    x = identity_block(x, 3, [256, 256, 768], stage=4, block="e2")
    conv4_6 = identity_block(x, 3, [256, 256, 768], stage=4, block="f2")
    # celui-là aussi

    cbcr = BatchNormalization(input_shape=input_shape_cbcr)(input_cbcr)
    cbcr = conv_block(cbcr, 1, [256, 256, 256], stage=2, block="a5", strides=(1, 1))
    x = Concatenate(axis=-1)([conv4_6, cbcr])

    return x, [28, 28, input_shape_y[2] + input_shape_cbcr[2]], input_y, input_cbcr


if __name__ == '__main__':
    # ResNet50Custom(weights=None)
    # ResNet50RGB(weights='imagenet')
    ResNet50Custom(weights='imagenet').summary()

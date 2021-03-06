import os
from six.moves import urllib
import tensorflow as tf

_FILE_PATTERN = '%s-*'

_SPLITS_TO_SIZES = {
    'train': 1281167,
    'validation': 50000,
}

def get_split(split_name, dataset_dir, file_pattern=None, reader=None):
    """Gets a dataset tuple with instructions for reading ImageNet.
    Args:
        split_name: A train/test split name.
        dataset_dir: The base directory of the dataset sources.
        file_pattern: The file pattern to use when matching the dataset sources.
        It is assumed that the pattern contains a '%s' string so that the split
        name can be inserted.
        reader: The TensorFlow reader type.
    Returns:
        A `Dataset` namedtuple.
    Raises:
        ValueError: if `split_name` is not a valid train/test split.
    """
    if split_name not in _SPLITS_TO_SIZES:
        raise ValueError('split name %s was not recognized.' % split_name)

    if not file_pattern:
        file_pattern = _FILE_PATTERN
    file_pattern = os.path.join(dataset_dir, file_pattern % split_name)

    # Allowing None in the signature so that dataset_factory can use the default.
    if reader is None:
        reader = tf.TFRecordReader

    keys_to_features = {
        'image/encoded': tf.FixedLenFeature(
            (), tf.string, default_value=''),
        'image/format': tf.FixedLenFeature(
            (), tf.string, default_value='jpeg'),
        'image/class/label': tf.FixedLenFeature(
            [], dtype=tf.int64, default_value=-1),
        'image/class/text': tf.FixedLenFeature(
            [], dtype=tf.string, default_value=''),
        'image/object/bbox/xmin': tf.VarLenFeature(
            dtype=tf.float32),
        'image/object/bbox/ymin': tf.VarLenFeature(
            dtype=tf.float32),
        'image/object/bbox/xmax': tf.VarLenFeature(
            dtype=tf.float32),
        'image/object/bbox/ymax': tf.VarLenFeature(
            dtype=tf.float32),
        'image/object/class/label': tf.VarLenFeature(
            dtype=tf.int64),
    }

    items_to_handlers = {
        'image': slim.tfexample_decoder.Image('image/encoded', 'image/format'),
        'label': slim.tfexample_decoder.Tensor('image/class/label'),
        'label_text': slim.tfexample_decoder.Tensor('image/class/text'),
        'object/bbox': slim.tfexample_decoder.BoundingBox(
            ['ymin', 'xmin', 'ymax', 'xmax'], 'image/object/bbox/'),
        'object/label': slim.tfexample_decoder.Tensor('image/object/class/label'),
    }

    decoder = slim.tfexample_decoder.TFExampleDecoder(
        keys_to_features, items_to_handlers)

    labels_to_names = None
    if LOAD_READABLE_NAMES:
        if dataset_utils.has_labels(dataset_dir):
            labels_to_names = dataset_utils.read_label_file(dataset_dir)
        else:
            labels_to_names = create_readable_names_for_imagenet_labels()
            dataset_utils.write_label_file(labels_to_names, dataset_dir)

    return slim.dataset.Dataset(
        data_sources=file_pattern,
        reader=reader,
        decoder=decoder,
        num_samples=_SPLITS_TO_SIZES[split_name],
        items_to_descriptions=_ITEMS_TO_DESCRIPTIONS,
        num_classes=_NUM_CLASSES,
        labels_to_names=labels_to_names)
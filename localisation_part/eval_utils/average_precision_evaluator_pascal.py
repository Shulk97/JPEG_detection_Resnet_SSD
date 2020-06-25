'''
An evaluator to compute the Pascal VOC-style mean average precision (both the pre-2010
and post-2010 algorithm versions) of a given Keras SSD model on a given dataset.

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
'''

from __future__ import division
import numpy as np
from math import ceil
from tqdm import trange
import sys
import warnings
import os

from data_generator.object_detection_2d_data_generator import DataGenerator
from data_generator.object_detection_2d_geometric_ops import Resize
from data_generator.object_detection_2d_patch_sampling_ops import RandomPadFixedAR
from data_generator.object_detection_2d_photometric_ops import ConvertTo3Channels
from ssd_encoder_decoder.ssd_output_decoder import decode_detections
from data_generator.object_detection_2d_misc_utils import apply_inverse_transforms

from bounding_box_utils.bounding_box_utils import iou

class Evaluator:
    '''
    Computes the mean average precision of the given Keras SSD model on the given dataset.

    Can compute the Pascal-VOC-style average precision in both the pre-2010 (k-point sampling)
    and post-2010 (integration) algorithm versions.

    Optionally also returns the average precisions, precisions, and recalls.

    The algorithm is identical to the official Pascal VOC pre-2010 detection evaluation algorithm
    in its default settings, but can be cusomized in a number of ways.
    '''

    def __init__(self,
                 n_classes,
                 data_generator,
                 predictions,
                 pred_format={'class_id': 0, 'conf': 1, 'xmin': 2, 'ymin': 3, 'xmax': 4, 'ymax': 5}):
        '''
        Arguments:
            n_classes (int): The number of positive classes, e.g. 20 for Pascal VOC, 80 for MS COCO.
            data_generator (DataGenerator): A `DataGenerator` object with the evaluation dataset.
            pred_format (dict, optional): A dictionary that defines which index in the last axis of the model's decoded predictions
                contains which bounding box coordinate. The dictionary must map the keywords 'class_id', 'conf' (for the confidence),
                'xmin', 'ymin', 'xmax', and 'ymax' to their respective indices within last axis.
            gt_format (list, optional): A dictionary that defines which index of a ground truth bounding box contains which of the five
                items class ID, xmin, ymin, xmax, ymax. The expected strings are 'xmin', 'ymin', 'xmax', 'ymax', 'class_id'.
        '''

        if not isinstance(data_generator, DataGenerator):
            warnings.warn("`data_generator` is not a `DataGenerator` object, which will cause undefined behavior.")

        self.data_generator = data_generator
        self.n_classes = n_classes
        self.pred_format = pred_format

        # The following lists all contain per-class data, i.e. all list have the length `n_classes + 1`,
        # where one element is for the background class, i.e. that element is just a dummy entry.
        self.prediction_results = None
        self.num_gt_per_class = None
        self.true_positives = None
        self.false_positives = None
        self.cumulative_true_positives = None
        self.cumulative_false_positives = None
        self.cumulative_precisions = None # "Cumulative" means that the i-th element in each list represents the precision for the first i highest condidence predictions for that class.
        self.cumulative_recalls = None # "Cumulative" means that the i-th element in each list represents the recall for the first i highest condidence predictions for that class.
        self.average_precisions = None
        self.mean_average_precision = None

    def __call__(self,
                 img_height,
                 img_width,
                 data_generator_mode='resize',
                 round_confidences=False,
                 matching_iou_threshold=0.5,
                 border_pixels='include',
                 sorting_algorithm='quicksort',
                 average_precision_mode='integrate',
                 num_recall_points=11,
                 ignore_neutral_boxes=True,
                 return_precisions=False,
                 return_recalls=False,
                 return_average_precisions=False,
                 verbose=True,
                 decoding_confidence_thresh=0.01,
                 decoding_iou_threshold=0.45,
                 decoding_top_k=200,
                 decoding_pred_coords='centroids',
                 decoding_normalize_coords=True):

        self.get_num_gt_per_class()

        self.match_predictions(ignore_neutral_boxes=ignore_neutral_boxes,
                               matching_iou_threshold=matching_iou_threshold,
                               border_pixels=border_pixels,
                               sorting_algorithm=sorting_algorithm,
                               verbose=verbose,
                               ret=False)

        self.compute_precision_recall(verbose=verbose, ret=False)

        self.compute_average_precisions(mode=average_precision_mode,
                                        num_recall_points=num_recall_points,
                                        verbose=verbose,
                                        ret=False)

        mean_average_precision = self.compute_mean_average_precision(ret=True)

        return mean_average_precision, self.average_precisions, self.precisions, self.recalls

    def get_num_gt_per_class(self):

        num_gt_per_class = np.zeros(shape=(self.n_classes+1), dtype=np.int)

        ground_truth = self.data_generator.labels

        print('Computing the number of positive ground truth boxes per class.')
        tr = trange(len(ground_truth), file=sys.stdout)

        # Iterate over the ground truth for all images in the dataset.
        for i in tr:
            boxes = self.data_generator.labels[i]
            boxes = np.asarray(boxes)

            # Iterate over all ground truth boxes for the current image.
            for j in range(boxes.shape[0]):

                # If there is no such thing as evaluation-neutral boxes for
                # our dataset, always increment the counter for the respective
                # class ID.
                class_id = boxes[j, 0]
                num_gt_per_class[class_id] += 1

        self.num_gt_per_class = num_gt_per_class

    def match_predictions(self,
                          ignore_neutral_boxes=True,
                          matching_iou_threshold=0.5,
                          border_pixels='include',
                          sorting_algorithm='quicksort',
                          verbose=True,
                          ret=False):

        # Convert the ground truth to a more efficient format for what we need
        # to do, which is access ground truth by image ID repeatedly.
        ground_truth = {}
        eval_neutral_available = not (self.data_generator.eval_neutral is None) # Whether or not we have annotations to decide whether ground truth boxes should be neutral or not.
        for i in range(len(self.data_generator.image_ids)):
            image_id = str(self.data_generator.image_ids[i])
            labels = self.data_generator.labels[i]

            ground_truth[image_id] = np.asarray(labels)

        true_positives = [[]] # The false positives for each class, sorted by descending confidence.
        false_positives = [[]] # The true positives for each class, sorted by descending confidence.
        cumulative_true_positives = [[]]
        cumulative_false_positives = [[]]

        # Iterate over all classes.
        for class_id in range(1, self.n_classes + 1):

            predictions = self.predictions[class_id]

            # Store the matching results in these lists:
            true_pos = np.zeros(len(predictions), dtype=np.int) # 1 for every prediction that is a true positive, 0 otherwise
            false_pos = np.zeros(len(predictions), dtype=np.int) # 1 for every prediction that is a false positive, 0 otherwise

            # In case there are no predictions at all for this class, we're done here.
            if len(predictions) == 0:
                print("No predictions for class {}/{}".format(class_id, self.n_classes))
                true_positives.append(true_pos)
                false_positives.append(false_pos)
                continue

            # Convert the predictions list for this class into a structured array so that we can sort it by confidence.

            # Get the number of characters needed to store the image ID strings in the structured array.
            num_chars_per_image_id = len(str(predictions[0][0])) + 6 # Keep a few characters buffer in case some image IDs are longer than others.
            # Create the data type for the structured array.
            preds_data_type = np.dtype([('image_id', 'U{}'.format(num_chars_per_image_id)),
                                        ('confidence', 'f4'),
                                        ('xmin', 'f4'),
                                        ('ymin', 'f4'),
                                        ('xmax', 'f4'),
                                        ('ymax', 'f4')])
            # Create the structured array
            predictions = np.array(predictions, dtype=preds_data_type)

            # Sort the detections by decreasing confidence.
            descending_indices = np.argsort(-predictions['confidence'], kind=sorting_algorithm)
            predictions_sorted = predictions[descending_indices]

            if verbose:
                tr = trange(len(predictions), file=sys.stdout)
                tr.set_description("Matching predictions to ground truth, class {}/{}.".format(class_id, self.n_classes))
            else:
                tr = range(len(predictions.shape))

            # Keep track of which ground truth boxes were already matched to a detection.
            gt_matched = {}

            # Iterate over all predictions.
            for i in tr:

                prediction = predictions_sorted[i]
                image_id = prediction['image_id']
                pred_box = np.asarray(list(prediction[['xmin', 'ymin', 'xmax', 'ymax']])) # Convert the structured array element to a regular array.

                # Get the relevant ground truth boxes for this prediction,
                # i.e. all ground truth boxes that match the prediction's
                # image ID and class ID.

                # The ground truth could either be a tuple with `(ground_truth_boxes, eval_neutral_boxes)`
                # or only `ground_truth_boxes`.
                if ignore_neutral_boxes and eval_neutral_available:
                    gt, eval_neutral = ground_truth[image_id]
                else:
                    gt = ground_truth[image_id]
                gt = np.asarray(gt)
                class_mask = gt[:,class_id_gt] == class_id
                gt = gt[class_mask]
                if ignore_neutral_boxes and eval_neutral_available:
                    eval_neutral = eval_neutral[class_mask]

                if gt.size == 0:
                    # If the image doesn't contain any objects of this class,
                    # the prediction becomes a false positive.
                    false_pos[i] = 1
                    continue

                # Compute the IoU of this prediction with all ground truth boxes of the same class.
                overlaps = iou(boxes1=gt[:,[xmin_gt, ymin_gt, xmax_gt, ymax_gt]],
                               boxes2=pred_box,
                               coords='corners',
                               mode='element-wise',
                               border_pixels=border_pixels)

                # For each detection, match the ground truth box with the highest overlap.
                # It's possible that the same ground truth box will be matched to multiple
                # detections.
                gt_match_index = np.argmax(overlaps)
                gt_match_overlap = overlaps[gt_match_index]

                if gt_match_overlap < matching_iou_threshold:
                    # False positive, IoU threshold violated:
                    # Those predictions whose matched overlap is below the threshold become
                    # false positives.
                    false_pos[i] = 1
                else:
                    if not (ignore_neutral_boxes and eval_neutral_available) or (eval_neutral[gt_match_index] == False):
                        # If this is not a ground truth that is supposed to be evaluation-neutral
                        # (i.e. should be skipped for the evaluation) or if we don't even have the
                        # concept of neutral boxes.
                        if not (image_id in gt_matched):
                            # True positive:
                            # If the matched ground truth box for this prediction hasn't been matched to a
                            # different prediction already, we have a true positive.
                            true_pos[i] = 1
                            gt_matched[image_id] = np.zeros(shape=(gt.shape[0]), dtype=np.bool)
                            gt_matched[image_id][gt_match_index] = True
                        elif not gt_matched[image_id][gt_match_index]:
                            # True positive:
                            # If the matched ground truth box for this prediction hasn't been matched to a
                            # different prediction already, we have a true positive.
                            true_pos[i] = 1
                            gt_matched[image_id][gt_match_index] = True
                        else:
                            # False positive, duplicate detection:
                            # If the matched ground truth box for this prediction has already been matched
                            # to a different prediction previously, it is a duplicate detection for an
                            # already detected object, which counts as a false positive.
                            false_pos[i] = 1

            true_positives.append(true_pos)
            false_positives.append(false_pos)

            cumulative_true_pos = np.cumsum(true_pos) # Cumulative sums of the true positives
            cumulative_false_pos = np.cumsum(false_pos) # Cumulative sums of the false positives

            cumulative_true_positives.append(cumulative_true_pos)
            cumulative_false_positives.append(cumulative_false_pos)

        self.true_positives = true_positives
        self.false_positives = false_positives
        self.cumulative_true_positives = cumulative_true_positives
        self.cumulative_false_positives = cumulative_false_positives

        if ret:
            return true_positives, false_positives, cumulative_true_positives, cumulative_false_positives

    def compute_precision_recall(self):

        cumulative_precisions = [[]]
        cumulative_recalls = [[]]

        # Iterate over all classes.
        for class_id in range(1, self.n_classes + 1):

            print("Computing precisions and recalls, class {}/{}".format(class_id, self.n_classes))

            tp = self.cumulative_true_positives[class_id]
            fp = self.cumulative_false_positives[class_id]


            cumulative_precision = np.where(tp + fp > 0, tp / (tp + fp), 0) # 1D array with shape `(num_predictions,)`
            cumulative_recall = tp / self.num_gt_per_class[class_id] # 1D array with shape `(num_predictions,)`

            cumulative_precisions.append(cumulative_precision)
            cumulative_recalls.append(cumulative_recall)

        self.cumulative_precisions = cumulative_precisions
        self.cumulative_recalls = cumulative_recalls



    def compute_average_precisions(self, mode='sample', num_recall_points=11):

        average_precisions = [0.0]

        # Iterate over all classes.
        for class_id in range(1, self.n_classes + 1):

            print("Computing average precision, class {}/{}".format(class_id, self.n_classes))

            cumulative_precision = self.cumulative_precisions[class_id]
            cumulative_recall = self.cumulative_recalls[class_id]
            average_precision = 0.0

            if mode == 'sample':

                for t in np.linspace(start=0, stop=1, num=num_recall_points, endpoint=True):

                    cum_prec_recall_greater_t = cumulative_precision[cumulative_recall >= t]

                    if cum_prec_recall_greater_t.size == 0:
                        precision = 0.0
                    else:
                        precision = np.amax(cum_prec_recall_greater_t)

                    average_precision += precision

                average_precision /= num_recall_points

            elif mode == 'integrate':

                # We will compute the precision at all unique recall values.
                unique_recalls, unique_recall_indices, unique_recall_counts = np.unique(cumulative_recall, return_index=True, return_counts=True)

                # Store the maximal precision for each recall value and the absolute difference
                # between any two unique recal values in the lists below. The products of these
                # two nummbers constitute the rectangular areas whose sum will be our numerical
                # integral.
                maximal_precisions = np.zeros_like(unique_recalls)
                recall_deltas = np.zeros_like(unique_recalls)

                # Iterate over all unique recall values in reverse order. This saves a lot of computation:
                # For each unique recall value `r`, we want to get the maximal precision value obtained
                # for any recall value `r* >= r`. Once we know the maximal precision for the last `k` recall
                # values after a given iteration, then in the next iteration, in order compute the maximal
                # precisions for the last `l > k` recall values, we only need to compute the maximal precision
                # for `l - k` recall values and then take the maximum between that and the previously computed
                # maximum instead of computing the maximum over all `l` values.
                # We skip the very last recall value, since the precision after between the last recall value
                # recall 1.0 is defined to be zero.
                for i in range(len(unique_recalls)-2, -1, -1):
                    begin = unique_recall_indices[i]
                    end   = unique_recall_indices[i + 1]
                    # When computing the maximal precisions, use the maximum of the previous iteration to
                    # avoid unnecessary repeated computation over the same precision values.
                    # The maximal precisions are the heights of the rectangle areas of our integral under
                    # the precision-recall curve.
                    maximal_precisions[i] = np.maximum(np.amax(cumulative_precision[begin:end]), maximal_precisions[i + 1])
                    # The differences between two adjacent recall values are the widths of our rectangle areas.
                    recall_deltas[i] = unique_recalls[i + 1] - unique_recalls[i]

                average_precision = np.sum(maximal_precisions * recall_deltas)

            average_precisions.append(average_precision)

        return average_precisions

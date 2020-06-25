import numpy as np

from tqdm import tqdm

def iou(box, boxes):
    """ Compute the Intersection Over Union between a box and a list of boxes.

    # Argument
        box: The box to be compared against. The box is a numpy array the following shape, [x1, y1, x2, y2], the two points are the top left and the bottom right corner, the order does not matter.
        boxes: The boxes to compare the box against. boxes is a numpy array of the following shape: (n_box, 4), as for the box argument, the order of the point for each box doesn't matter, it just have to be the same in both cases.

    # Return
        A list of iou, one for each box of the boxes argument.
    """

    # Compute the intersection
    intersection_up_left = np.maximum(boxes[:, :2], box[:2])
    intersection_bottom_right = np.minimum(boxes[:, 2:4], box[2:])

    intersection_wh = intersection_bottom_right - intersection_up_left + 1

    # If no intersection
    intersection_wh = np.maximum(intersection_wh, 0)

    intersection = intersection_wh[:, 0] * intersection_wh[:, 1]

    # Compute union
    area_box = (box[2] - box[0] + 1) * (box[3] - box[1] + 1)

    area_boxes = (boxes[:, 2] - boxes[:, 0] + 1) * (boxes[:, 3] - boxes[:, 1] + 1)

    union = area_box + area_boxes - intersection

    # Compute iou
    return intersection / union

def match_boxes(predictions, ground_truth, iou_threshold, ignore_under_area=None):
    # Creating the matrix that will hold the result data

    if predictions.size == 0:
        return []

    # Sorting the prediction by confidence level (confidence level > iou when
    # associating the boxes)

    predictions = predictions[predictions[:, 0].argsort()]
    predictions = predictions[::-1]

    results = [[0, predictions[i, 0]] for i in range(len(predictions))]

    # If no box to match, everything is considered as false positives
    if (ground_truth.size == 0):
        return results

    # Array to keep track of the already assigned boxes
    assigned = np.zeros(len(ground_truth))

    # For each prediction, we try to match a box (confidence level > iou when
    # associating the boxes)
    for b in range(len(predictions)):
        # If all the boxes are matched, everything else is false positive
        if np.all(assigned):
            break

        # Computing the IoUs
        current_box = predictions[b, 1:]

        ious = iou(current_box, ground_truth[:,:-1])
        # Searching for the best iou
        iou_idx = np.argmax(ious)

        if ious[iou_idx] < iou_threshold:
            continue

        # If not assigned we set the box as match and exit the loop
        if not assigned[iou_idx]:
            assigned[iou_idx] = True
            if ground_truth[iou_idx][-1] == 1:
                results[b][0] = 2
            elif ignore_under_area is not None:
                if ((ground_truth[iou_idx][2] - ground_truth[iou_idx][0]) * (ground_truth[iou_idx][3] - ground_truth[iou_idx][1])) < ignore_under_area:
                    results[b][0] = 2
                else:
                    results[b][0] = 1
            else:
                results[b][0] = 1


    return results

def compute_true_false_positives(predictions, ground_truth, num_classes,ignore_under_area=None):

    print("Matching the boxes for each image.")
    true_false_positives = {new_list: [] for new_list in range(1, num_classes + 1)}
    ground_truth_per_classes = {new_list: 0 for new_list in range(1, num_classes + 1)}

    for image_number, boxes in enumerate(tqdm(predictions)):
        class_boxes = {new_list: [] for new_list in range(1, num_classes + 1)}
        class_groundtruth = {new_list: [] for new_list in range(1, num_classes + 1)}
        class_groundtruth_no_difficult = {new_list: [] for new_list in range(1, num_classes + 1)}

        for box in boxes:
            class_boxes[box[0]].append(box[1:])

        for box in ground_truth[image_number]:
            class_groundtruth[box[0]].append(box[1:])
            if box[-1] == 0:
                if ignore_under_area is not None:
                    if ((box[3] - box[1]) * (box[4] - box[2])) >= ignore_under_area:
                        class_groundtruth_no_difficult[box[0]].append(box[1:])
                else:
                    class_groundtruth_no_difficult[box[0]].append(box[1:])

        for class_id in range(1,num_classes + 1):
            results = match_boxes(np.array(class_boxes[class_id]), np.array
            (class_groundtruth[class_id]), 0.5, ignore_under_area=ignore_under_area)

            if results: 
                true_false_positives[class_id] = true_false_positives[class_id] + results
            ground_truth_per_classes[class_id] += len(class_groundtruth_no_difficult[class_id])

    # Sort the tp fp at each iteration
    for key in true_false_positives:
        if true_false_positives[key]:
            true_false_positives[key] = sorted(true_false_positives[key], key=lambda l: l[1], reverse=True)

    return true_false_positives, ground_truth_per_classes

def compute_recall_precision(true_false_positive, ground_truth_number):
    # Setting the results matrix
    recall = np.zeros(len(true_false_positive))
    precision = np.zeros(len(true_false_positive))

    if ground_truth_number == 0:
        return recall, precision

    # Setting the number of True Positives already seen.
    diff = 0
    positives_sum = 0.
    for i, value in enumerate(true_false_positive):
        if value[0] == 2:
            diff += 1
        else:
            positives_sum += value[0]
            recall[i] = positives_sum / ground_truth_number
            precision[i] = positives_sum / (i + 1. - diff)

    return recall, precision

def compute_average_precision_sample(recall, precision, num_recall_points):
    average_precision = 0.0

    recall = np.array(recall)
    precision = np.array(precision)

    for i in np.linspace(start=0, stop=1, num=num_recall_points, endpoint=True):

        indices = recall >= i
        if np.any(indices):
            precision_i = np.max(precision[indices])
        else:
            precision_i = 0.0

        average_precision += precision_i

    return average_precision / num_recall_points


def compute_average_precision_integrate(recall, precision):

    recall = np.array(recall)
    precision = np.array(precision)


    unique_recalls, unique_recall_indices, unique_recall_counts = np.unique(recall, return_index=True, return_counts=True)

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
    for i in range(len(unique_recalls)-2, 0, -1):
        begin = unique_recall_indices[i]
        end   = unique_recall_indices[i + 1]
        # When computing the maximal precisions, use the maximum of the previous iteration to
        # avoid unnecessary repeated computation over the same precision values.
        # The maximal precisions are the heights of the rectangle areas of our integral under
        # the precision-recall curve.
        print("-------")
        print(begin)
        print(end)
        print(precision[begin:end])
        print(maximal_precisions[i + 1])
        print(np.amax(precision[begin:end]))
        maximal_precisions[i] = np.maximum(np.amax(precision[begin:end]), maximal_precisions[i + 1])
        # The differences between two adjacent recall values are the widths of our rectangle areas.
        recall_deltas[i] = unique_recalls[i + 1] - unique_recalls[i]

    average_precision = np.sum(maximal_precisions * recall_deltas)

    return average_precision


def compute_average_precisions(predictions, ground_truth, num_classes, mode="sample", num_recall_points=11, ignore_under_area=None):

    average_precisions = []

    print("Matching all the predictions to the ground truth boxes for each image.")
    true_false_positives, ground_truth_per_classes = compute_true_false_positives(predictions, ground_truth, num_classes, ignore_under_area=ignore_under_area)

    print("Processing the mAP in {} mode.".format(mode))
    for class_number in range(1, num_classes + 1):

        recall, precision = compute_recall_precision(true_false_positives[class_number], ground_truth_per_classes[class_number])

        if mode == "sample":
            average_precisions.append(compute_average_precision_sample(recall, precision, num_recall_points))
        elif mode == "integrate":
            average_precisions.append(compute_average_precision_integrate(recall, precision))

    return average_precisions
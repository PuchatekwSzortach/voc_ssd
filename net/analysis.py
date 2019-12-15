"""
Script with analysis code
"""

import collections

import matplotlib.pyplot as plt
import numpy as np
import seaborn
import tqdm
import vlogging

import net.ssd
import net.utilities


def is_annotation_matched(annotation, match_candidates):
    """
    Check if annotation is matched by any predictions. Match is called if there is a prediction
    with same category as annotation and their intersection over union is above 0.5
    :param annotation: net.utilities.Annotation instance
    :param match_candidates: list of net.utilities.Annotation instances
    :return: bool
    """

    # Pick up bounding boxes for all predictions that have same category as ground truth annotations
    bounding_boxes = \
        [match_candidate.bounding_box for match_candidate in match_candidates if
         match_candidate.label == annotation.label]

    # If no candidate has correct category, then there is no match
    if len(bounding_boxes) == 0:

        return False

    # Return indices of boxes that have intersection over union with annotation's box that's over 0.5
    matched_default_boxes_indices = net.utilities.get_matched_boxes_indices(
        annotation.bounding_box, np.array(bounding_boxes))

    # We have a match if matched_default_boxes_indices list is non-empty
    return len(matched_default_boxes_indices) > 0


class MatchingDataComputer:
    """
    Utility for computing matched and unmatched annotations and predictions at different thresholds
    """

    def __init__(self, samples_loader, model, default_boxes_factory, thresholds, categories):
        """
        Constructor
        :param samples_loader: net.data.VOCSamplesDataLoader instance
        :param model: net.ml.VGGishModel instance
        :param default_boxes_factory: net.data.VOCSamplesDataLoader instance
        :param thresholds: list of floats, for each threshold, only predictions with confidence above it will be used
        to compute matching data
        :param categories: list of strings, labels for categories
        """

        self.samples_loader = samples_loader
        self.model = model
        self.default_boxes_factory = default_boxes_factory
        self.thresholds = thresholds
        self.categories = categories

    def get_thresholds_matched_data_map(self):
        """
        Runs predictions over all samples from samples model and returns matches data for each threshold.
        :return: dictionary {float: matching data} for each threshold.
        matching data is a dictionary with keys:
        matched_annotations, unmatched_annotations, matched_predictions, unmatched_predictions
        """

        iterator = iter(self.samples_loader)

        thresholds_matched_data_map = {threshold: collections.defaultdict(list) for threshold in self.thresholds}

        for _ in tqdm.tqdm(range(len(self.samples_loader))):

            image, ground_truth_annotations = next(iterator)

            default_boxes_matrix = self.default_boxes_factory.get_default_boxes_matrix(image.shape)
            softmax_predictions_matrix = self.model.predict(image)

            for threshold in self.thresholds:

                matches_data = thresholds_matched_data_map[threshold]

                predicted_annotations = net.ssd.get_predicted_annotations(
                    default_boxes_matrix=default_boxes_matrix,
                    softmax_predictions_matrix=softmax_predictions_matrix,
                    categories=self.categories,
                    threshold=threshold)

                # For each ground truth annotation, check if it was matched by any prediction
                for ground_truth_annotation in ground_truth_annotations:

                    if is_annotation_matched(ground_truth_annotation, predicted_annotations):

                        matches_data["matched_annotations"].append(ground_truth_annotation)

                    else:

                        matches_data["unmatched_annotations"].append(ground_truth_annotation)

                # For each prediction, check if it was matched by any ground truth annotation
                for prediction in predicted_annotations:

                    if is_annotation_matched(prediction, ground_truth_annotations):

                        matches_data["matched_predictions"].append(prediction)

                    else:

                        matches_data["unmatched_predictions"].append(prediction)

        return thresholds_matched_data_map


def get_precision_recall_analysis_report(
        matched_annotations, unmatched_annotations, matched_predictions, unmatched_predictions):
    """
    Get report string for precision recall analysis
    :param matched_annotations: list of net.data.Annotation instances, annotations that had matching predictions
    :param unmatched_annotations:
    list of net.data.Annotation instances, annotations that didn't have matching predictions
    :param matched_predictions: list of net.data.Annotation instances,
    predictions that had matching ground truth annotations
    :param unmatched_predictions: list of net.data.Annotation instances,
    annotations that didn't have matching predictions
    :return: str
    """
    messages = []

    total_annotations_count = len(matched_annotations) + len(unmatched_annotations)
    recall = len(matched_annotations) / total_annotations_count if total_annotations_count > 0 else 0

    message = "Recall is {:.3f}<br>".format(recall)
    messages.append(message)

    total_predictions_count = len(matched_predictions) + len(unmatched_predictions)
    precision = len(matched_predictions) / total_predictions_count if total_predictions_count > 0 else 0

    message = "Precision is {:.3f}<br>".format(precision)
    messages.append(message)

    return " ".join(messages)


def log_precision_recall_analysis(logger, thresholds_matching_data_map):
    """
    Log precision recall analysis
    :param logger: logger instance
    :param thresholds_matching_data_map: dictionary, each key is a float and value is a dictionary with
    info about matched and unmatched annotations and predictions at corresponding threshold
    """

    for threshold in sorted(thresholds_matching_data_map.keys()):

        logger.info("At threshold {}<br>".format(threshold))

        matching_data = thresholds_matching_data_map[threshold]

        report = get_precision_recall_analysis_report(
            matched_annotations=matching_data["matched_annotations"],
            unmatched_annotations=matching_data["unmatched_annotations"],
            matched_predictions=matching_data["matched_predictions"],
            unmatched_predictions=matching_data["unmatched_predictions"])

        logger.info(report)
        logger.info("<br>")


def get_heatmap(data, bin_size, max_size):
    """
    Given a 2D data, compute its heatmap.
    Heatmap bins data into max_size // step bins, with all data falling outside of heatmap's max_size excluded
    :param data: 2D numpy array
    :param bin_size: int, size of a single bin
    :param max_size: size of the largest bin
    :return: 2D array
    """

    # Bin annotations into our loosely defined bins
    bins_count = max_size // bin_size

    heatmap = np.zeros(shape=(bins_count, bins_count), dtype=np.int32)

    coordinates = data // bin_size

    coordinates_of_data_points_within_max_size = coordinates[np.all(coordinates < bins_count, axis=1)]

    for y, x in coordinates_of_data_points_within_max_size:

        heatmap[y, x] += 1

    return heatmap


def get_annotations_sizes_heatmap_figure(annotations, bin_size, max_size):
    """
    Get a figure of annotations sizes heatmap
    :param annotations: list of net.utilities.Annotation instances
    :param bin_size: int, step at which sizes should be binned
    :param max_size: int max size of heatmap. Anything above it will be binned to largest bin
    :return: matplotlib pyplot figure instance
    """

    data = np.array([(annotation.height, annotation.width) for annotation in annotations])
    annotations_heatmap = net.analysis.get_heatmap(data=data, bin_size=bin_size, max_size=max_size)

    figure = plt.figure()
    seaborn.heatmap(data=annotations_heatmap)

    bins_count = max_size // bin_size
    original_ticks = np.arange(bins_count)
    overwritten_ticks = bin_size * np.arange(bins_count)

    plt.xticks(original_ticks, overwritten_ticks, rotation=90)
    plt.yticks(original_ticks, overwritten_ticks, rotation=0)

    figure.tight_layout()
    return figure


def log_performance_with_annotations_size_analysis(logger, thresholds_matching_data_map):
    """
    Log performance of network across annotations of different sizes
    :param logger: logger instance
    :param thresholds_matching_data_map: dictionary, each key is a float and value is a dictionary with
    info about matched and unmatched annotations and predictions at corresponding threshold
    """

    # Only use data at threshold 0
    unmatched_annotations = thresholds_matching_data_map[0]["unmatched_annotations"]

    unmatched_annotations_heatmap_figure = get_annotations_sizes_heatmap_figure(
        annotations=unmatched_annotations,
        bin_size=20,
        max_size=500)

    logger.info(vlogging.VisualRecord(
        "Unmatched annotations heatmap", unmatched_annotations_heatmap_figure))

    small_unmatched_annotations_heatmap_figure = get_annotations_sizes_heatmap_figure(
        annotations=unmatched_annotations,
        bin_size=5,
        max_size=100)

    logger.info(vlogging.VisualRecord(
        "Small unmatched annotations heatmap", small_unmatched_annotations_heatmap_figure))

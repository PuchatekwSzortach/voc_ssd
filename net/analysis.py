"""
Script with analysis code
"""

import collections

import numpy as np
import tqdm

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

    message = "Recall is {:.3f}<br>".format(
        len(matched_annotations) / (len(matched_annotations) + len(unmatched_annotations)))

    messages.append(message)

    message = "Precision is {:.3f}<br>".format(
        len(matched_predictions) / (len(matched_predictions) + len(unmatched_predictions)))

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

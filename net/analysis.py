"""
Script with analysis code
"""

import collections
import os
import queue
import threading

import matplotlib.pyplot as plt
import numpy as np
import seaborn
import tqdm
import vlogging
import xmltodict

import net.data
import net.ssd
import net.utilities


def get_filtered_dataset_annotations(config):
    """
    Retrieves annotations for the dataset, scales them in accordance to how their images would be scaled
    in prediction, filters out unusually sized annotations, then returns annotations that made it through filtering
    :param config: configuration dictionary
    :return: list of net.utilities.Annotation instances
    """

    images_filenames = net.data.get_dataset_filenames(
        config["voc"]["data_directory"], config["voc"]["validation_set_path"])

    annotations_paths = [os.path.join(config["voc"]["data_directory"], "Annotations", image_filename + ".xml")
                         for image_filename in images_filenames]

    labels_to_categories_index_map = {label: index for (index, label) in enumerate(config["categories"])}

    all_annotations = []

    for annotations_path in tqdm.tqdm(annotations_paths):

        with open(annotations_path, encoding="utf-8") as file:

            image_annotations_xml = xmltodict.parse(file.read())

            image_size = \
                int(image_annotations_xml["annotation"]["size"]["height"]), \
                int(image_annotations_xml["annotation"]["size"]["width"])

            # Read annotations
            annotations = net.data.get_objects_annotations(
                image_annotations=image_annotations_xml,
                labels_to_categories_index_map=labels_to_categories_index_map)

            # Resize annotations in line with how we would resize the image
            annotations = [annotation.resize(image_size, config["size_factor"]) for annotation in annotations]

            all_annotations.extend(annotations)

    return all_annotations


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
        annotation.bounding_box, np.array(bounding_boxes), threshold=0.5)

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

        samples_count = len(self.samples_loader)
        samples_queue = queue.Queue(maxsize=250)

        computations_thread = threading.Thread(
            target=self._matching_computations,
            args=(thresholds_matched_data_map, samples_queue, samples_count))

        computations_thread.start()

        for _ in range(samples_count):

            image, ground_truth_annotations = next(iterator)
            softmax_predictions_matrix, offsets_predictions_matrix = self.model.predict(image)

            sample_data_map = {
                "image_shape": image.shape,
                "ground_truth_annotations": ground_truth_annotations,
                "softmax_predictions_matrix": softmax_predictions_matrix,
                "offsets_predictions_matrix": offsets_predictions_matrix
            }

            # Put data on a queue, matching computations thread will process that data and put results into
            # thresholds_matched_data_map
            samples_queue.put(sample_data_map)

        samples_queue.join()
        computations_thread.join()

        return thresholds_matched_data_map

    def _matching_computations(self, thresholds_matched_data_map, samples_data_queue, samples_count):

        for _ in tqdm.tqdm(range(samples_count)):

            # Get sample data from queue
            sample_data_map = samples_data_queue.get()

            samples_data_queue.task_done()

            default_boxes_matrix = self.default_boxes_factory.get_default_boxes_matrix(sample_data_map["image_shape"])

            # Compute matching data for sample at each threshold
            for threshold in self.thresholds:

                predictions = net.ssd.PredictionsComputer(
                    categories=self.categories,
                    threshold=threshold,
                    use_non_maximum_suppression=True).get_predictions(
                        bounding_boxes_matrix=default_boxes_matrix + sample_data_map["offsets_predictions_matrix"],
                        softmax_predictions_matrix=sample_data_map["softmax_predictions_matrix"])

                matches_data_for_single_sample = self._get_matches_data(
                    ground_truth_annotations=sample_data_map["ground_truth_annotations"],
                    predictions=predictions)

                matches_data = thresholds_matched_data_map[threshold]

                # Add computed matched data for current sample to stored data for all samples
                for key, value in matches_data_for_single_sample.items():

                    matches_data[key].extend(value)

    @staticmethod
    def _get_matches_data(ground_truth_annotations, predictions):

        matches_data = collections.defaultdict(list)

        # For each ground truth annotation, check if it was matched by any prediction
        for ground_truth_annotation in ground_truth_annotations:

            if is_annotation_matched(ground_truth_annotation, predictions):

                matches_data["matched_annotations"].append(ground_truth_annotation)

            else:

                matches_data["unmatched_annotations"].append(ground_truth_annotation)

        # For each prediction, check if it was matched by any ground truth annotation
        for prediction in predictions:

            if is_annotation_matched(prediction, ground_truth_annotations):

                matches_data["matched_predictions"].append(prediction)

            else:

                matches_data["unmatched_predictions"].append(prediction)

        matches_data["mean_average_precision_data"] = get_predictions_matches(
            ground_truth_annotations=ground_truth_annotations, predictions=predictions)

        return matches_data


def get_precision_recall_analysis_report(
        matched_annotations, unmatched_annotations, matched_predictions, unmatched_predictions):
    """
    Get report string for precision recall analysis
    :param matched_annotations: list of net.data.Annotation instances,
    ground truth annotations that had matching predictions
    :param unmatched_annotations: list of net.data.Annotation instances,
    ground truth annotations that didn't have matching predictions
    :param matched_predictions: list of net.data.Annotation instances,
    predictions that had matching ground truth annotations
    :param unmatched_predictions: list of net.data.Annotation instances,
    predictions that didn't have matching ground truth annotations
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


def log_unmatched_annotations_sizes(
        logger, unmatched_annotations, x_range, y_range, size_factor, instances_threshold):
    """
    Log statistics about unmatched annotations sizes within specified x and y ranges, binned by specified size_factor.
    Only objects missed more than instances threshold are reported
    :param logger: logger instance
    :param unmatched_annotations: list of net.utilities.Annotation instances,
    :param x_range: tuple of ints (min_x_size, max_x_size), lower value is inclusive
    :param y_range: tuple of ints (min_y_size, max_y_size), lower value is inclusive
    :param size_factor: int, size factor within which objects are binned together
    :param instances_threshold: int, threshold above which missed object count for given size must be for that
    size to be reported
    """

    def is_annotation_within_target_size(annotation):
        return x_range[0] <= annotation.width < x_range[1] and y_range[0] <= annotation.height < y_range[1]

    unmatched_annotations_within_target_size = \
        [annotation for annotation in unmatched_annotations if is_annotation_within_target_size(annotation) is True]

    misses_counts_sizes_tuples = net.utilities.get_objects_sizes_analysis(
        unmatched_annotations_within_target_size, size_factor=size_factor)

    thresholded_counts_sizes_tuples = \
        [element for element in misses_counts_sizes_tuples if element[0] > instances_threshold]

    large_unmatched_annotations_analysis_message = "<br>".join([
        "{} -> {}".format(count, size) for count, size in thresholded_counts_sizes_tuples])

    header_template = (
        "<h3>Unmatched annotations in y range {} and x range {}, binned by size factor {} "
        "and thresholded above {}:</h3>"
    ).format(y_range, x_range, size_factor, instances_threshold)

    logger.info(header_template + large_unmatched_annotations_analysis_message)


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

    log_unmatched_annotations_sizes(
        logger=logger,
        unmatched_annotations=unmatched_annotations,
        x_range=(100, 600),
        y_range=(100, 600),
        size_factor=20,
        instances_threshold=10)

    log_unmatched_annotations_sizes(
        logger=logger,
        unmatched_annotations=unmatched_annotations,
        x_range=(40, 200),
        y_range=(40, 200),
        size_factor=10,
        instances_threshold=10)

    log_unmatched_annotations_sizes(
        logger=logger,
        unmatched_annotations=unmatched_annotations,
        x_range=(10, 100),
        y_range=(10, 100),
        size_factor=5,
        instances_threshold=10)


def get_predictions_matches(ground_truth_annotations, predictions):
    """
    Get predictions matches - for each prediction return a dictionary with:
    - prediction: net.utilities.Prediction instance
    - is_correct: bool - whether prediction matches any ground truth annotation
    Results are arranged by predictions' confidences, in descending order
    This function computes predictions matches VOC 2007 mean average precision style - that is
    ground truth annotation can be matched by only one prediction - so all lower-confidence predictions
    that have high IOU with that ground truth annotation are counted as failed.
    Match is counted if IOU with ground truth annotation is above 0.5.
    :param ground_truth_annotations: list of net.utilities.Annotation instances, ground truth annotations
    :param predictions: list of net.utilities.Prediction instances
    :return: list of dictionaries, one for each prediction, sorted by prediction confidence
    """

    sorted_predictions = sorted(predictions, key=lambda x: x.confidence, reverse=True)

    # Set of ground truth annotations that weren't matched with any prediction yet
    unmatched_ground_truth_annotations = set(ground_truth_annotations)

    matches_data = []

    for prediction in sorted_predictions:

        # Convert set of unmatched ground truth annotations to a list, so we can work with its indices
        unmatched_ground_truth_annotations_list = list(unmatched_ground_truth_annotations)

        if len(unmatched_ground_truth_annotations_list) > 0:

            # Get a boolean vector checking if prediction the same label as any ground truth annotations
            categories_matches_vector = [ground_truth_annotation.label == prediction.label
                                         for ground_truth_annotation in unmatched_ground_truth_annotations_list]

            annotations_bounding_boxes = np.array([
                ground_truth_annotation.bounding_box
                for ground_truth_annotation in unmatched_ground_truth_annotations_list
            ])

            # Return indices of ground truth annotation's boxes that have high intersection over union with
            # prediction's box
            matched_boxes_indices = net.utilities.get_matched_boxes_indices(
                prediction.bounding_box, annotations_bounding_boxes, threshold=0.5)

            # Create boxes matches vector
            boxes_matches_vector = np.zeros_like(categories_matches_vector)
            boxes_matches_vector[matched_boxes_indices] = True

            # Create matches vector by doing logical and on categories and boxes vectors
            matches_flags_vector = np.logical_and(categories_matches_vector, boxes_matches_vector)

            # Record match data for the prediction
            matches_data.append(
                {
                    "prediction": prediction,
                    "is_correct": bool(np.any(matches_flags_vector))
                }
            )

            # Remove matched ground truth annotations from unmatched ground truth annotations set
            unmatched_ground_truth_annotations = unmatched_ground_truth_annotations.difference(
                np.array(unmatched_ground_truth_annotations_list)[matches_flags_vector])

        else:

            matches_data.append(
                {
                    "prediction": prediction,
                    "is_correct": False
                }
            )

    return matches_data


def log_mean_average_precision_analysis(logger, thresholds_matching_data_map):
    """
    Log VOC Pascal 2007 style mean average precision for predictions across different thresholds
    :param logger: logger instance
    :param thresholds_matching_data_map: dictionary, each key is a float and value is a dictionary with
    info about predictions matches
    """

    for threshold in sorted(thresholds_matching_data_map.keys()):

        logger.info("At threshold {}<br>".format(threshold))

        matching_data = thresholds_matching_data_map[threshold]

        ground_truth_annotations_count = \
            len(matching_data["matched_annotations"]) + len(matching_data["unmatched_annotations"])

        mean_average_precision = MeanAveragePrecisionComputer.get_mean_average_precision(
            predictions_matches_data=matching_data["mean_average_precision_data"],
            ground_truth_annotations_count=ground_truth_annotations_count)

        logger.info("Mean average precision is: {:.4f}".format(mean_average_precision))
        logger.info("<br>")


class MeanAveragePrecisionComputer:
    """
    Class for computing VOC Pascal 2007 style mean average precision.
    Based on notes from https://medium.com/@jonathan_hui/map-mean-average-precision-for-object-detection-45c121a31173
    """

    @staticmethod
    def get_recall_values_and_precision_values_data(predictions_matches_data, ground_truth_annotations_count):
        """
        Given predictions matches data return recall values and precision values computed up to each prediction
        matches data element
        :param predictions_matches_data: list of dictionaries with prediction and information whether it matched
        a ground truth annotation
        :param ground_truth_annotations_count: int, number of ground truth annotations in dataset
        :return: tuple of two lists, first contains recall values across predictions matches data,
        second contains precision values across predictions matches data
        """

        sorted_predictions_matches_data = sorted(
            predictions_matches_data, key=lambda x: x["prediction"].confidence, reverse=True)

        predictions_correctness_values = []

        for prediction_match_data in sorted_predictions_matches_data:

            predictions_correctness_values.append(int(prediction_match_data["is_correct"]))

        predictions_correctness_cumulative_sums = np.cumsum(predictions_correctness_values)

        # Progressive recall values from first prediction till last
        recall_values = predictions_correctness_cumulative_sums / ground_truth_annotations_count

        # Progressive precision values from first prediction till last
        precision_values = \
            predictions_correctness_cumulative_sums / np.arange(1, len(predictions_correctness_values) + 1)

        return recall_values, precision_values

    @staticmethod
    def get_mean_average_precision(predictions_matches_data, ground_truth_annotations_count):
        """
        Get VOC Pascal 2007 style mean average precision
        :param predictions_matches_data: list of dictionaries, each contains prediction and information whether
        it matched any ground truth annotation
        :param ground_truth_annotations_count: int, number of ground truth annotations
        :return: float
        """

        recall_values, precision_values = \
            MeanAveragePrecisionComputer.get_recall_values_and_precision_values_data(
                predictions_matches_data=predictions_matches_data,
                ground_truth_annotations_count=ground_truth_annotations_count)

        smoothed_out_precision_values = MeanAveragePrecisionComputer.get_smoothed_out_precision_values(
            precision_values)

        interpolated_precision_values = MeanAveragePrecisionComputer.get_interpolated_precision_values(
            recall_values=recall_values, precision_values=smoothed_out_precision_values)

        return np.mean(interpolated_precision_values)

    @staticmethod
    def get_smoothed_out_precision_values(precision_values):
        """
        Given progressive precision values for predictions from most to least confident,
        smooth them out VOC Pascal 2007 style.
        For each precision value replace it with the maximum value to its right.
        :param precision_values: 1D numpy array of floats
        :return: 1D numpy array of floats
        """

        # Flip precision values - numpy offers function that computes cumulative max from left to right,
        # but we want this behaviour from right to left. The easiest solution is to flip array,
        # compute cumulative maximum, then flip it back.
        flipped_precision_values = np.flip(precision_values, axis=0)
        cumulative_maximum_values = np.maximum.accumulate(flipped_precision_values)

        return np.flip(cumulative_maximum_values, axis=0)

    @staticmethod
    def get_interpolated_precision_values(recall_values, precision_values):
        """
        Given recall values and precision values, return precision values at recall values from 0 to 1 in 0.1 step.
        If some recall values are never reached, we set interpolated precision values for them to 0.
        :param recall_values: 1D numpy array of floats
        :param precision_values: 1D numpy array floats
        :return: 11-elements long 1D numpy array of floats,
        """

        # If we don't have recall values going all the way up to 0.9999, add recall values at 0.00001 above
        # max value and at 1, set their respective precision values to 0.
        # This way interpolation in this range will return precision of 0.
        if recall_values[-1] < 0.9999:

            recall_values = np.concatenate((recall_values, [recall_values[-1] + 0.0001, 1]))
            precision_values = np.concatenate((precision_values, [0, 0]))

        interpolation_coordinates = np.arange(start=0, stop=1.1, step=0.1)

        interpolated_precision_values = np.interp(
            x=interpolation_coordinates,
            xp=recall_values,
            fp=precision_values)

        return interpolated_precision_values


def get_mean_losses(model, ssd_model_configuration, samples_loader):
    """
    Get means losses for network across dataset
    :param model: net.data.VOCSamplesDataLoader instance
    :param ssd_model_configuration: dictionary with model configuration options
    :param samples_loader: net.data.VOCSamplesDataLoader instance
    :return: dictionary with mean losses values
    """

    ssd_samples_loader = net.ssd.SSDTrainingLoopDataLoader(
        voc_samples_data_loader=samples_loader,
        ssd_model_configuration=ssd_model_configuration)

    losses_map = {
        "total": [],
        "categorical": [],
        "offset": []
    }

    iterator = iter(ssd_samples_loader)

    for _ in tqdm.tqdm(range(len(ssd_samples_loader))):

        image, default_boxes_categories_ids_vector, default_boxes_sizes, ground_truth_offsets = \
            next(iterator)

        feed_dictionary = {
            model.network.input_placeholder: np.array([image]),
            model.ops_map["default_boxes_categories_ids_vector_placeholder"]: default_boxes_categories_ids_vector,
            model.ops_map["default_boxes_sizes_op"]: default_boxes_sizes,
            model.ops_map["ground_truth_offsets_matrix_op"]: ground_truth_offsets
        }

        total_loss, categorical_loss, offsets_loss = model.session.run(
            [model.ops_map["loss_op"], model.ops_map["categorical_loss_op"], model.ops_map["offsets_loss_op"]],
            feed_dictionary)

        losses_map["total"].append(total_loss)
        losses_map["categorical"].append(categorical_loss)
        losses_map["offset"].append(offsets_loss)

    return {key: np.mean(value) for key, value in losses_map.items()}

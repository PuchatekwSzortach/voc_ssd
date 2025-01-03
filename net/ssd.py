"""
Module with SSD-specific computations
"""

import box
import numpy as np
import tensorflow as tf

import net.utilities


class DefaultBoxesFactory:
    """
    Class for generating default boxes
    """

    def __init__(self, model_configuration):
        """
        Constructor
        :param model_configuration: dictionary with SSD model configuration options
        """

        self.model_configuration = model_configuration

        self.cache = {}

    def get_default_boxes_matrix(self, image_shape):
        """
        Gets default boxes matrix for whole SSD model - made out of concatenations of
        default boxes matrices for different heads
        :param image_shape: tuple of ints, with first two elements being image's height and width
        :return: 2D numpy array
        """

        # If no cached matrix is present, compute one and store it in cache
        if image_shape not in self.cache.keys():

            default_boxes_matrices = []

            for prediction_head in self.model_configuration["prediction_heads_order"]:

                single_head_default_boxes_matrix = self.get_default_boxes_matrix_for_single_prediction_head(
                    self.model_configuration[prediction_head], image_shape)

                default_boxes_matrices.append(single_head_default_boxes_matrix)

            self.cache[image_shape] = np.concatenate(default_boxes_matrices)

        # Serve matrix from cache
        return self.cache[image_shape]

    @staticmethod
    def get_default_boxes_matrix_for_single_prediction_head(configuration, image_shape):
        """
        Gets default boxes matrix for a single prediction head
        :param configuration: dictionary of configuration options to be used to make default boxes matrix
        :param image_shape: tuple of ints, with first two elements being image's height and width
        :return: 2D numpy array
        """

        step = configuration["image_downscale_factor"]

        boxes_matrices = []

        # Boxes must be laid out in (y, x, boxes configurations) order, since that's how
        # convolutional network will lay them out in memory
        for y_center in np.arange(step // 2, image_shape[0], step):

            for x_center in np.arange(step // 2, image_shape[1], step):

                boxes_matrix = DefaultBoxesFactory.get_boxes_at_location(
                    y_center=y_center,
                    x_center=x_center,
                    configuration=configuration)

                boxes_matrices.append(boxes_matrix)

        return np.concatenate(boxes_matrices)

    @staticmethod
    def get_boxes_at_location(y_center, x_center, configuration):
        """
        Get all boxes variations at specified center location
        :param y_center: int
        :param x_center: int
        :param configuration: dictionary with configuration options
        :return:
        """

        boxes = []

        for base_size in configuration["base_bounding_box_sizes"]:

            # Vertical boxes
            for aspect_ratio in configuration["aspect_ratios"]:

                width = aspect_ratio * base_size
                height = base_size

                half_width = width / 2
                half_height = height / 2

                corner_box = [
                    x_center - half_width,
                    y_center - half_height,
                    x_center + half_width,
                    y_center + half_height
                ]

                boxes.append(corner_box)

            # Horizontal boxes
            for aspect_ratio in configuration["aspect_ratios"]:

                width = base_size
                height = aspect_ratio * base_size

                half_width = width / 2
                half_height = height / 2

                corner_box = [
                    x_center - half_width,
                    y_center - half_height,
                    x_center + half_width,
                    y_center + half_height
                ]

                boxes.append(corner_box)

        return np.array(boxes)


def get_matching_analysis_generator(ssd_model_configuration, ssd_input_generator, threshold):
    """
    Generator that accepts ssd_input_generator and yield a generator that outputs tuples of
    matched_annotations and unmatched_annotations, both of which are lists of Annotation instances.
    :param ssd_model_configuration: dictionary of options specifying ssd model's configuration
    :param ssd_input_generator: generator that outputs (image, annotations) tuples
    :param threshold: float, threshold above which box must overlap with ground truth annotation
    to be counted as a match
    :return: generator that outputs (matched_annotations, unmatched_annotations) tuples
    """

    default_boxes_factory = DefaultBoxesFactory(ssd_model_configuration)

    while True:

        image, annotations = next(ssd_input_generator)
        default_boxes_matrix = default_boxes_factory.get_default_boxes_matrix(image.shape)

        matched_annotations = []
        unmatched_annotations = []

        for annotation in annotations:

            matched_default_boxes_indices = net.utilities.get_matched_boxes_indices(
                annotation.bounding_box, default_boxes_matrix, threshold)

            if len(matched_default_boxes_indices) > 0:

                matched_annotations.append(annotation)

            else:

                unmatched_annotations.append(annotation)

        yield matched_annotations, unmatched_annotations


class SSDTrainingLoopDataLoader:
    """
    Data loader class that outputs tuples
    image, default_boxes_categories_ids_vector, default_boxes_sizes, offsets)
    or data suitable for training and evaluating SSD network
    """

    def __init__(self, voc_samples_data_loader, ssd_model_configuration):
        """
        Constructor
        :param voc_samples_data_loader: net.data.VOCSamplesDataLoader instance
        :param ssd_model_configuration: map with model configuration
        """

        self.voc_samples_data_loader = voc_samples_data_loader
        self.default_boxes_factory = DefaultBoxesFactory(ssd_model_configuration)

    def __len__(self):
        return len(self.voc_samples_data_loader)

    def __iter__(self):

        iterator = iter(self.voc_samples_data_loader)

        while True:

            image, annotations = next(iterator)
            default_boxes_matrix = self.default_boxes_factory.get_default_boxes_matrix(image.shape)

            matched_default_boxes_indices = []
            matched_default_boxes_categories_ids = []

            offsets = []

            # For each annotation collect indices of default boxes that were matched,
            # matched categories indices and offsets from ground truth boxes to default boxes
            for annotation in annotations:

                training_data = self._get_training_data_for_single_annotation(annotation, default_boxes_matrix)

                matched_default_boxes_indices.extend(training_data["matched_default_boxes_indices"])
                matched_default_boxes_categories_ids.extend(training_data["matched_default_boxes_categories_ids"])
                offsets.extend(training_data["offsets"])

            # Create a vector for all default boxes and set values to categories boxes were matched with
            default_boxes_categories_ids_vector = np.zeros(shape=default_boxes_matrix.shape[0], dtype=np.int32)
            offsets_matrix = np.zeros(shape=(default_boxes_matrix.shape[0], 4), dtype=np.float32)

            if len(matched_default_boxes_indices) > 0:

                default_boxes_categories_ids_vector[matched_default_boxes_indices] = \
                    matched_default_boxes_categories_ids

                offsets_matrix[matched_default_boxes_indices] = offsets

            default_boxes_sizes = np.array([
                default_boxes_matrix[:, 2] - default_boxes_matrix[:, 0],
                default_boxes_matrix[:, 3] - default_boxes_matrix[:, 1]
            ]).transpose()

            yield image, default_boxes_categories_ids_vector, default_boxes_sizes, offsets_matrix

    @staticmethod
    def _get_training_data_for_single_annotation(annotation, default_boxes_matrix):

        matched_default_boxes_indices = net.utilities.get_matched_boxes_indices(
            annotation.bounding_box, default_boxes_matrix, threshold=0.5)

        matched_default_boxes_categories_ids = [annotation.category_id] * len(matched_default_boxes_indices)

        offsets = [
            annotation.bounding_box - default_box
            for default_box in default_boxes_matrix[matched_default_boxes_indices]]

        return {
            "matched_default_boxes_indices": matched_default_boxes_indices,
            "matched_default_boxes_categories_ids": matched_default_boxes_categories_ids,
            "offsets": offsets
        }


class PredictionsComputer:
    """
    Class for computing objects predictions from predictions matrix and default boxes matrix
    """

    def __init__(self, categories, confidence_threshold, post_processing_config: box.Box):
        """
        Constructor
        :param categories: list of strings
        :param threshold: float, only non-background predictions above this threshold will be returned
        :param post_processing_config: box.Box with post-processing options. Should contain key
        "method". Other keys depend on method chosen.
        """

        self.categories = categories
        self.confidence_threshold = confidence_threshold
        self.post_processing_config = post_processing_config

    def get_predictions(self, bounding_boxes_matrix, softmax_predictions_matrix):
        """
        Get list of predicted annotations based on default boxes matrix and softmax predictions matrix
        :param bounding_boxes_matrix: 2D numpy array, each row represents coordinates of a single bounding box
        :param softmax_predictions_matrix: 2D numpy array, each row represents one-hot encoded softmax predictions
        for a corresponding bounding box
        :return: list of net.utilities.Prediction instances
        """

        if self.post_processing_config.non_maximum_suppression.method is None:

            return self._get_raw_predictions(bounding_boxes_matrix, softmax_predictions_matrix)

        else:

            return self._get_nms_predictions(bounding_boxes_matrix, softmax_predictions_matrix)

    def _get_raw_predictions(self, bounding_boxes_matrix, softmax_predictions_matrix):

        # Get a selector for non-background predictions over threshold
        predictions_selector = \
            (np.argmax(softmax_predictions_matrix, axis=1) > 0) & \
            (np.max(softmax_predictions_matrix, axis=1) > self.confidence_threshold)

        predictions_boxes = bounding_boxes_matrix[predictions_selector]
        predictions_categories_indices = np.argmax(softmax_predictions_matrix[predictions_selector], axis=1)
        predictions_confidences = np.max(softmax_predictions_matrix[predictions_selector], axis=1)

        predictions = []

        for prediction_box, category_id, confidence in \
                zip(predictions_boxes, predictions_categories_indices, predictions_confidences):

            prediction = net.utilities.Prediction(
                bounding_box=[int(x) for x in prediction_box],
                confidence=confidence,
                label=self.categories[category_id],
                category_id=category_id)

            predictions.append(prediction)

        return predictions

    def _get_nms_predictions(self, default_boxes_matrix, softmax_predictions_matrix):

        # Get a selector for non-background predictions over threshold
        predictions_selector = \
            (np.argmax(softmax_predictions_matrix, axis=1) > 0) & \
            (np.max(softmax_predictions_matrix, axis=1) > self.confidence_threshold)

        predictions_boxes = default_boxes_matrix[predictions_selector]
        predictions_categories_indices = np.argmax(softmax_predictions_matrix[predictions_selector], axis=1)

        predictions = []

        # Get predictions scores as a column vector
        predictions_scores = np.max(softmax_predictions_matrix[predictions_selector], axis=1).reshape(-1, 1)

        # Merge predictions boxes and scores together into detections matrix
        detections = np.concatenate([predictions_boxes, predictions_scores], axis=1)

        # soft nms works on each category separately
        for category_id in range(1, len(self.categories)):

            if self.post_processing_config.non_maximum_suppression.method == "soft":

                # Perform soft-nms on detections for current category
                retained_detections_at_current_category = \
                    net.utilities.get_detections_after_soft_non_maximum_suppression(
                        detections=detections[predictions_categories_indices == category_id],
                        sigma=self.post_processing_config.non_maximum_suppression.sigma,
                        score_threshold=self.post_processing_config.non_maximum_suppression.score_threshold)

            elif self.post_processing_config.non_maximum_suppression.method == "greedy":

                retained_detections_at_current_category = \
                    net.utilities.get_detections_after_greedy_non_maximum_suppression(
                        detections=detections[predictions_categories_indices == category_id],
                        iou_threshold=self.post_processing_config.non_maximum_suppression.iou_threshold)

            else:
                raise ValueError(
                    "Unsupported non-maximum suppression method: "
                    f"{self.post_processing_config.non_maximum_suppression.method}"
                )

            for detection in retained_detections_at_current_category:

                prediction = net.utilities.Prediction(
                    bounding_box=[int(x) for x in detection[:4]],
                    confidence=detection[4],
                    label=self.categories[category_id],
                    category_id=category_id)

                predictions.append(prediction)

        return predictions


class SingleShotDetectorLossBuilder:
    """
    Class for building SSD loss op
    """

    def __init__(
            self, default_boxes_categories_ids_vector_op, categories_predictions_logits_matrix_op,
            hard_negatives_mining_ratio, default_boxes_sizes_op,
            ground_truth_offsets_matrix_op, offsets_predictions_matrix_op):
        """
        Constructor
        :param default_boxes_categories_ids_vector_op: tensorflow tensor with shape (None,) and int type
        :param categories_predictions_logits_matrix_op: tensorflow tensor with shape (None, None) and float type
        :param hard_negatives_mining_ratio: int - specifies ratio of hard negatives to positive samples
        categorical loss op should use
        :param default_boxes_sizes_op: tensorflow tensor with shape (None, 2) and int type,
        each row represents width and height of corresponding default box
        :param ground_truth_offsets_matrix_op: tensorflow tensor with shape (None, 4) and float type,
        each row represents correct offsets from ground truth annotations to default boxes.
        If default box wasn't matched to any annotation, its row will be all zeros.
        :param: localizations_offsets_predictions_matrix_op: tensorflow tensor with shape (None, 4) and float type,
        network predictions for offsets from default boxes to ground truth boxes
        """

        self.ops_map = {
            "default_boxes_categories_ids_vector_op": default_boxes_categories_ids_vector_op,
            "predictions_logits_matrix_op": categories_predictions_logits_matrix_op,
            "default_boxes_sizes_op": default_boxes_sizes_op,
            "ground_truth_offsets_matrix_op": ground_truth_offsets_matrix_op,
            "offsets_predictions_matrix_op": offsets_predictions_matrix_op
        }

        self.hard_negatives_mining_ratio = hard_negatives_mining_ratio

        default_boxes_count = tf.shape(self.ops_map["default_boxes_categories_ids_vector_op"])[0]

        all_ones_op = tf.ones(shape=(default_boxes_count,), dtype=tf.float32)
        all_zeros_op = tf.zeros(shape=(default_boxes_count,), dtype=tf.float32)

        # Get a selector that's set to 1 for boxes that are matched with ground truth annotations
        # and to zero elsewhere
        positive_matches_selector_op = tf.where(
            self.ops_map["default_boxes_categories_ids_vector_op"] > 0, all_ones_op, all_zeros_op)

        positive_matches_count_op = tf.cast(tf.reduce_sum(positive_matches_selector_op), tf.int32)

        self.categorical_loss_op = self._build_categorical_loss_op(
            positive_matches_selector_op, positive_matches_count_op)

        self.offsets_loss_op = self._build_offsets_loss(
            positive_matches_selector_op, positive_matches_count_op)

        self.loss_op = (self.categorical_loss_op + self.offsets_loss_op) / 2.0

    def _build_categorical_loss_op(self, positive_matches_selector_op, positive_matches_count_op):

        raw_loss_op = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=self.ops_map["default_boxes_categories_ids_vector_op"],
            logits=self.ops_map["predictions_logits_matrix_op"])

        # Get positive losses op - that is op with losses only for default bounding boxes
        # that were matched with ground truth annotations.
        # First multiply raw losses with selector op, so that losses for all boxes matched with
        # background will be set zero.
        # Then sort losses in descending order and select positive_matches_count elements.
        # Thus end effect is that we select positive losses only
        positive_losses_op = tf.sort(
            raw_loss_op * positive_matches_selector_op, direction='DESCENDING')[:positive_matches_count_op]

        # Get negative losses op, that is op with losses for default boxes that weren't matched with any ground truth
        # annotations, or should predict background, in a similar manner as we did for positive losses.
        # Choose x times positive matches count largest losses only for hard negatives mining
        negative_losses_op = tf.sort(
            raw_loss_op * (1.0 - positive_matches_selector_op),
            direction='DESCENDING')[:(self.hard_negatives_mining_ratio * positive_matches_count_op)]

        # If there were any positive matches at all, then return mean of both losses.
        # Otherwise return 0 - as we can't have a mean of an empty op.
        return tf.cond(
            pred=positive_matches_count_op > 0,
            true_fn=lambda: tf.math.reduce_mean(tf.concat(values=[positive_losses_op, negative_losses_op], axis=0)),
            false_fn=lambda: tf.constant(0, dtype=tf.float32))

    def _build_offsets_loss(self, positive_matches_selector_op, positive_matches_count_op):

        # Get error between ground truth offsets and predicted offsets
        offsets_errors_op = \
            self.ops_map["ground_truth_offsets_matrix_op"] - self.ops_map["offsets_predictions_matrix_op"]

        float_boxes_sizes_op = tf.cast(self.ops_map["default_boxes_sizes_op"], tf.float32)

        # Scale errors by box width for x-offsets and box height for y-offsets, so their values
        # are roughly within <-1, 1> scale
        scaled_offsets_errors_op = tf.stack([
            offsets_errors_op[:, 0] / float_boxes_sizes_op[:, 0],
            offsets_errors_op[:, 1] / float_boxes_sizes_op[:, 1],
            offsets_errors_op[:, 2] / float_boxes_sizes_op[:, 0],
            offsets_errors_op[:, 3] / float_boxes_sizes_op[:, 1]
            ], axis=1)

        # Square errors to get positive values, compute mean value per box
        raw_losses_op = tf.reduce_mean(tf.math.pow(scaled_offsets_errors_op, 2), axis=1)

        # Multiply by matches selector, so that we only compute loss at default boxes that matched ground truth
        # annotations, then select all these losses
        positive_losses_op = tf.sort(
            raw_losses_op * positive_matches_selector_op, direction='DESCENDING')[:positive_matches_count_op]

        # # And finally return mean value of positives losses, or 0 if there were none
        return tf.cond(
            pred=positive_matches_count_op > 0,
            true_fn=lambda: tf.reduce_mean(positive_losses_op),
            false_fn=lambda: tf.constant(0, dtype=tf.float32))


def get_matched_default_boxes(annotations, default_boxes_matrix):
    """
    Get default boxes that matched annotations
    :param annotations: list of Annotation instances
    :param default_boxes_matrix: 2D numpy array of default boxes
    :return: 2D numpy array of default boxes
    """

    # Get default boxes matches
    all_matched_default_boxes_indices = []

    for annotation in annotations:

        matched_default_boxes_indices = net.utilities.get_matched_boxes_indices(
            annotation.bounding_box, default_boxes_matrix, threshold=0.5)

        all_matched_default_boxes_indices.extend(matched_default_boxes_indices.tolist())

    return default_boxes_matrix[all_matched_default_boxes_indices]

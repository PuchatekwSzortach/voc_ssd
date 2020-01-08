"""
Module with SSD-specific computations
"""

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

                box = [x_center - half_width, y_center - half_height, x_center + half_width, y_center + half_height]
                boxes.append(box)

            # Horizontal boxes
            for aspect_ratio in configuration["aspect_ratios"]:

                width = base_size
                height = aspect_ratio * base_size

                half_width = width / 2
                half_height = height / 2

                box = [x_center - half_width, y_center - half_height, x_center + half_width, y_center + half_height]
                boxes.append(box)

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
    (image, indices of default boxes that matched annotations in image, categories ids boxes matched),
    or data suitable for training and evaluating SSD network
    """

    def __init__(self, voc_samples_data_loader, ssd_model_configuration):
        """
        Constructor
        :param voc_samples_data_loader: net.data.VOCSamplesDataLoader instance
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

            all_matched_default_boxes_indices = []
            all_matched_default_boxes_categories_ids = []

            # For each annotation collect indices of default boxes that were matched,
            # as well as matched categories indices
            for annotation in annotations:

                # Get matched boxes indices. Use a slightly higher iou threshold than 0.5 to encourage only
                # confident matches
                matched_default_boxes_indices = net.utilities.get_matched_boxes_indices(
                    annotation.bounding_box, default_boxes_matrix, threshold=0.6)

                all_matched_default_boxes_indices.extend(matched_default_boxes_indices)

                matched_default_boxes_categories_ids = [annotation.category_id] * len(matched_default_boxes_indices)
                all_matched_default_boxes_categories_ids.extend(matched_default_boxes_categories_ids)

            # Create a vector for all default boxes and set values to categories boxes were matched with
            default_boxes_categories_ids_vector = np.zeros(shape=default_boxes_matrix.shape[0], dtype=np.int32)

            default_boxes_categories_ids_vector[all_matched_default_boxes_indices] = \
                all_matched_default_boxes_categories_ids

            yield image, default_boxes_categories_ids_vector


class PredictionsComputer:
    """
    Class for computing objects predictions from predictions matrix and default boxes matrix
    """

    def __init__(self, categories, threshold, use_non_maximum_suppression):
        """
        Constructor
        :param categories: list of strings
        :param threshold: float, only non-background predictions above this threshold will be returned
        :param use_non_maximum_suppression: bool, specifies if non maximum suppression should be used.
        soft-nms algorithm is used for non maximum suppression.
        """

        self.categories = categories
        self.threshold = threshold
        self.use_non_maximum_suppression = use_non_maximum_suppression

    def get_predictions(self, default_boxes_matrix, softmax_predictions_matrix):
        """
        Get list of predicted annotations based on default boxes matrix and softmax predictions matrix
        :param default_boxes_matrix: 2D numpy array, each row represents coordinates of a default bounding box
        :param softmax_predictions_matrix: 2D numpy array, each row represents one-hot encoded softmax predictions
        for a corresponding default box
        :return: list of net.utilities.Prediction instances
        """

        if self.use_non_maximum_suppression is True:

            return self._get_soft_nms_predictions(default_boxes_matrix, softmax_predictions_matrix)

        else:

            return self._get_raw_predictions(default_boxes_matrix, softmax_predictions_matrix)

    def _get_raw_predictions(self, default_boxes_matrix, softmax_predictions_matrix):

        # Get a selector for non-background predictions over threshold
        predictions_selector = \
            (np.argmax(softmax_predictions_matrix, axis=1) > 0) & \
            (np.max(softmax_predictions_matrix, axis=1) > self.threshold)

        predictions_boxes = default_boxes_matrix[predictions_selector]
        predictions_categories_indices = np.argmax(softmax_predictions_matrix[predictions_selector], axis=1)
        predictions_confidences = np.max(softmax_predictions_matrix[predictions_selector], axis=1)

        predictions = []

        for box, category_id, confidence in \
                zip(predictions_boxes, predictions_categories_indices, predictions_confidences):

            prediction = net.utilities.Prediction(
                bounding_box=[int(x) for x in box],
                confidence=confidence,
                label=self.categories[category_id],
                category_id=category_id)

            predictions.append(prediction)

        return predictions

    def _get_soft_nms_predictions(self, default_boxes_matrix, softmax_predictions_matrix):

        # Get a selector for non-background predictions over threshold
        predictions_selector = \
            (np.argmax(softmax_predictions_matrix, axis=1) > 0) & \
            (np.max(softmax_predictions_matrix, axis=1) > self.threshold)

        predictions_boxes = default_boxes_matrix[predictions_selector]
        predictions_categories_indices = np.argmax(softmax_predictions_matrix[predictions_selector], axis=1)

        predictions = []

        # Get predictions scores as a column vector
        predictions_scores = np.max(softmax_predictions_matrix[predictions_selector], axis=1).reshape(-1, 1)

        # Merge predictions boxes and scores together into detections matrix
        detections = np.concatenate([predictions_boxes, predictions_scores], axis=1)

        # soft nms works on each category separately
        for category_id in range(1, len(self.categories)):

            # Perform soft-nms on detections for current category
            retained_detections_at_current_category = net.utilities.get_detections_after_soft_non_maximum_suppression(
                detections=detections[predictions_categories_indices == category_id],
                sigma=0.5,
                score_threshold=0.7)

            for detection in retained_detections_at_current_category:

                prediction = net.utilities.Prediction(
                    bounding_box=[int(x) for x in detection[:4]],
                    confidence=detection[4],
                    label=self.categories[category_id],
                    category_id=category_id)

                predictions.append(prediction)

        return predictions


def get_single_shot_detector_loss_op(
        default_boxes_categories_ids_vector_op, predictions_logits_matrix_op, hard_negatives_mining_ratio):
    """
    Function to create single shot detector loss op
    :param default_boxes_categories_ids_vector_op: tensorflow tensor with shape (None,) and int type
    :param predictions_logits_matrix_op: tensorflow tensor with shape (None, None) and float type
    :param hard_negatives_mining_ratio: int - specifies ratio of hard negatives to positive samples loss op
    should use
    :return: scalar loss op
    """

    default_boxes_count = tf.shape(default_boxes_categories_ids_vector_op)[0]

    raw_loss_op = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=default_boxes_categories_ids_vector_op, logits=predictions_logits_matrix_op)

    all_ones_op = tf.ones(shape=(default_boxes_count,), dtype=tf.float32)
    all_zeros_op = tf.zeros(shape=(default_boxes_count,), dtype=tf.float32)

    # Get a selector that's set to 1 where for all positive losses, split positive losses and negatives losses
    positive_losses_selector_op = tf.where(
        default_boxes_categories_ids_vector_op > 0, all_ones_op, all_zeros_op)

    positive_matches_count_op = tf.cast(tf.reduce_sum(positive_losses_selector_op), tf.int32)

    # Get positive losses op - that is op with losses only for default bounding boxes
    # that were matched with ground truth annotations.
    # First multiply raw losses with selector op, so that all negative losses will be zero.
    # Then sort losses in descending order and select positive_matches_count elements.
    # Thus end effect is that we select positive losses only
    positive_losses_op = tf.sort(
        raw_loss_op * positive_losses_selector_op, direction='DESCENDING')[:positive_matches_count_op]

    # Get negative losses op that is op with losses for default boxes that weren't matched with any ground truth
    # annotations, or should predict background, in a similar manner as we did for positive losses.
    # Choose x times positive matches count largest losses only for hard negatives mining
    negative_losses_op = tf.sort(
        raw_loss_op * (1.0 - positive_losses_selector_op),
        direction='DESCENDING')[:(hard_negatives_mining_ratio * positive_matches_count_op)]

    # If there were any positive matches at all, then return mean of both losses.
    # Otherwise return 0 - as we can't have a mean of an empty op.
    return tf.cond(
        pred=positive_matches_count_op > 0,
        true_fn=lambda: tf.reduce_mean(tf.concat([positive_losses_op, negative_losses_op], axis=0)),
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

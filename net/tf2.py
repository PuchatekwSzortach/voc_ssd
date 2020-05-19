"""
Code with tensorflow 2.x code
"""

import numpy as np
import tensorflow as tf


class TF2TrainingLoopDataLoader:
    """
    Data loader for tensorflow 2.x based code
    """

    def __init__(self, ssd_training_loop_data_loader):
        """
        Constructor
        :param ssd_training_loop_data_loader: net.ssd.SSDTrainingLoopDataLoader instance
        """

        self.ssd_training_loop_data_loader = ssd_training_loop_data_loader

    def __len__(self):

        return len(self.ssd_training_loop_data_loader)

    def __iter__(self):

        iterator = iter(self.ssd_training_loop_data_loader)

        while True:

            image, default_boxes_categories_ids_vector, default_boxes_sizes, offsets_matrix = next(iterator)

            # categories predictions head needs only default_boxes_categories_ids_vector to compute loss,
            # but offsets predictions head needs also offsets matrix and default boxes sizes.
            # All of that data has to be rolled up into one big matrix, though
            labels_data = {
                "categories_predictions_head": np.array([default_boxes_categories_ids_vector]),
                "offsets_predictions_head": np.array(
                    [np.concatenate(
                        [default_boxes_categories_ids_vector.reshape(-1, 1), offsets_matrix, default_boxes_sizes],
                        axis=1)])
            }

            yield np.array([image]), labels_data


class BaseSSDNetwork:
    """
    Base abstract class for SSD network.
    Provides logic for building prediction heads, but expects subclasses to provide code
    that will select base layers for prediction heads
    """

    def __init__(self, model_configuration, categories_count):
        """
        Constructor, build a prediction network based on data expected to be provided by subclasses
        :param model_configuration: dictionary with model configuration options
        :param categories_count: int, number of output categories
        """

        self.model_configuration = model_configuration

        # Subclass should provide implementation of ops map
        base_layers_tensor_map = self.get_base_layers_tensors_map()

        self.input_placeholder = base_layers_tensor_map["input_placeholder"]

        self.ops_map = base_layers_tensor_map

        categories_predictions_heads_ops_list = []
        offset_predictions_heads_ops_list = []

        for block in model_configuration["prediction_heads_order"]:

            categories_predictions_head, offsets_predictions_head = self.get_predictions_head(
                input_op=base_layers_tensor_map[block],
                categories_count=categories_count,
                head_configuration=model_configuration[block])

            categories_predictions_heads_ops_list.append(categories_predictions_head)
            offset_predictions_heads_ops_list.append(offsets_predictions_head)

        self.batch_of_categories_predictions_logits_matrices_op = \
            tf.concat(categories_predictions_heads_ops_list, axis=1)

        self.batch_of_softmax_categories_predictions_matrices_op = tf.nn.softmax(
            logits=self.batch_of_categories_predictions_logits_matrices_op,
            axis=-1,
            name="categories_predictions_head")

        self.batch_of_offsets_predictions_matrices_op = tf.concat(
            values=offset_predictions_heads_ops_list,
            axis=1,
            name="offsets_predictions_head")

        self.model = tf.keras.models.Model(
            inputs=self.input_placeholder,
            outputs={
                "categories_predictions_head": self.batch_of_softmax_categories_predictions_matrices_op,
                "offsets_predictions_head": self.batch_of_offsets_predictions_matrices_op
            }
        )

        self.model.compile(
            optimizer='adam',
            loss={
                "categories_predictions_head": categories_predictions_loss,
                "offsets_predictions_head": offsets_predictions_loss
            })

    @staticmethod
    def get_predictions_head(input_op, categories_count, head_configuration):
        """
        Create a prediction head from input op given head configuration. Prediction head contains two prediction ops,
        one for categories and one for offsets.
        :param input_op: input tensor
        :param categories_count: int, number of filters output of prediction head should have - basically
        we want prediction head to predict a one-hot encoding for each default box
        :param head_configuration: dictionary with head configuration options
        :return: tuple (categories logits predictions op, offsets predictions op)
        """

        # Common part for both categories predictions and offsets predictions ops
        x = tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation=tf.nn.swish)(input_op)
        x = tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation=tf.nn.swish)(x)
        x = tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation=tf.nn.swish)(x)

        # Compute total number of default boxes prediction head should make prediction for.
        # For each pixel prediction head receives it should make predictions for
        # base bounding boxes sizes count * 2 * aspect ratios count default boxes centered on that pixel.
        # We multiple aspect rations by 2, since we compute them both in horizontal and vertical orientation
        default_boxes_count = \
            len(head_configuration["base_bounding_box_sizes"]) * \
            2 * len(head_configuration["aspect_ratios"])

        # For each default box we want to make one hot encoded prediction across all categories
        categories_logits_predictions_op = tf.keras.layers.Conv2D(
            filters=default_boxes_count * categories_count, kernel_size=(3, 3), padding='same')(x)

        # For each default box we want to make 4 localization predictions - [x_left, y_top, x_right, y_bottom]
        offsets_predictions_op = tf.keras.layers.Conv2D(
            filters=4 * default_boxes_count, kernel_size=(3, 3), padding='same')(x)

        # Reshape outputs to 3D matrices (batch_dimension, default boxes on all pixel locations, x), where
        # x is categories_count for categories logits predictions op and 4 for offsets predictions op
        return \
            tf.reshape(categories_logits_predictions_op,
                       shape=(tf.shape(categories_logits_predictions_op)[0], -1, categories_count)), \
            tf.reshape(offsets_predictions_op,
                       shape=(tf.shape(offsets_predictions_op)[0], -1, 4))

    def get_base_layers_tensors_map(self):
        """
        Hook for subclasses to provide base layers map.
        It should contain input placeholder tensor, as well as a tensor for every prediction head
        that network's configuration specifies
        :return: dictionary
        """

        raise NotImplementedError()


class VGGishNetwork(BaseSSDNetwork):
    """
    SSD model based on VGG
    """

    def __init__(self, model_configuration, categories_count):
        """
        Constructor
        :param model_configuration: dictionary with model configuration
        :param categories_count: number of categories to predict, including background
        """

        super().__init__(
            model_configuration=model_configuration,
            categories_count=categories_count
        )

    def get_base_layers_tensors_map(self):
        """
        Implementation of hook for base layers tensors.
        Provides an input placeholder tensor and tensors for bases of prediction heads
        :return: dictionary
        """

        network = tf.keras.applications.VGG16(include_top=False)

        tensors_map = {prediction_head: network.get_layer(prediction_head).output
                       for prediction_head in self.model_configuration["prediction_heads_order"]}

        tensors_map["input_placeholder"] = network.input

        return tensors_map


def categories_predictions_loss(labels_data, predictions_data):
    """
    SSD categories predictions loss
    """

    default_boxes_categories_ids_vector_op = tf.reshape(labels_data, shape=(-1,))

    default_boxes_count = tf.shape(default_boxes_categories_ids_vector_op)[0]
    categories_predictions_matrix = tf.reshape(predictions_data, shape=(default_boxes_count, -1))

    raw_loss_op = tf.keras.backend.sparse_categorical_crossentropy(
        target=default_boxes_categories_ids_vector_op,
        output=categories_predictions_matrix,
        from_logits=False,
        axis=-1)

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

    hard_negatives_mining_ratio = 3

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
        true_fn=lambda: tf.math.reduce_mean(tf.concat(values=[positive_losses_op, negative_losses_op], axis=0)),
        false_fn=lambda: tf.constant(0, dtype=tf.float32))


def offsets_predictions_loss(labels_data, predictions_data):
    """
    SSD offsets predictions loss
    """

    # Each row of labels data should contain
    # - one value for default bounding box category
    # - four values for offsets predictions
    # - two values for default boxes sizes
    labels_data_matrix = tf.reshape(labels_data, shape=(-1, 7))

    # Each row of predictions data should contain four values for four default box corners
    prediction_data_matrix = tf.reshape(predictions_data, shape=(-1, 4))

    # Split labels data matrix into components
    default_boxes_categories_ids_vector_op = labels_data_matrix[:, 0]
    ground_truth_offsets_matrix_op = labels_data_matrix[:, 1:5]
    default_boxes_sizes_op = labels_data_matrix[:, 5:7]

    default_boxes_count = tf.shape(labels_data_matrix)[0]

    all_ones_op = tf.ones(shape=(default_boxes_count,), dtype=tf.float32)
    all_zeros_op = tf.zeros(shape=(default_boxes_count,), dtype=tf.float32)

    # Get a selector that's set to 1 where for all positive losses, split positive losses and negatives losses
    positive_matches_selector_op = tf.where(
        default_boxes_categories_ids_vector_op > 0, all_ones_op, all_zeros_op)

    positive_matches_count_op = tf.cast(tf.reduce_sum(positive_matches_selector_op), tf.int32)

    offsets_errors_op = ground_truth_offsets_matrix_op - prediction_data_matrix

    float_boxes_sizes_op = tf.cast(default_boxes_sizes_op, tf.float32)

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

    # And finally return mean value of positives losses, or 0 if there were none
    return tf.cond(
        pred=positive_matches_count_op > 0,
        true_fn=lambda: tf.reduce_mean(positive_losses_op),
        false_fn=lambda: tf.constant(0, dtype=tf.float32))


class HistoryLogger(tf.keras.callbacks.Callback):
    """
    Callback that training history
    """

    def __init__(self, logger):
        """
        Constructor
        :param logger: logging.Logger instance
        """

        super().__init__()

        self.logger = logger

    def on_epoch_end(self, epoch, logs=None):
        """
        Callback called on epoch end
        """

        self.logger.info(logs)

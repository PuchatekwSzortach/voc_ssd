"""
Module with machine learning code
"""

import os

import numpy as np
import tensorflow as tf
import tqdm


class VGGishNetwork:
    """
    SSD model based on VGG
    """

    def __init__(self, model_configuration, categories_count):
        """
        Constructor
        :param model_configuration: dictionary with model configuration
        :param categories_count: number of categories to predict, including background
        """

        self.model_configuration = model_configuration

        vgg = tf.keras.applications.VGG16(include_top=False)

        self.input_placeholder = vgg.input

        self.ops_map = {
            "block2_pool": vgg.get_layer("block2_pool").output,
            "block3_pool": vgg.get_layer("block3_pool").output,
            "block4_pool": vgg.get_layer("block4_pool").output,
            "block5_pool": vgg.get_layer("block5_pool").output,
        }

        self.prediction_heads = {
            "block2_head": self.get_prediction_head(
                input_op=self.ops_map["block2_pool"],
                categories_count=categories_count,
                head_configuration=model_configuration["block2_head"]),
            "block3_head": self.get_prediction_head(
                input_op=self.ops_map["block3_pool"],
                categories_count=categories_count,
                head_configuration=model_configuration["block3_head"]),
            "block4_head": self.get_prediction_head(
                input_op=self.ops_map["block4_pool"],
                categories_count=categories_count,
                head_configuration=model_configuration["block4_head"]),
            "block5_head": self.get_prediction_head(
                input_op=self.ops_map["block5_pool"],
                categories_count=categories_count,
                head_configuration=model_configuration["block5_head"]),
        }

        # Create prediction logits of assembling all prediction heads into a single matrix
        predictions_heads_ops_list = \
            [self.prediction_heads[name] for name in model_configuration["prediction_heads_order"]]

        self.batch_of_predictions_logits_matrices_op = tf.concat(predictions_heads_ops_list, axis=1)

    @staticmethod
    def get_prediction_head(input_op, categories_count, head_configuration):
        """
        Creates a prediction head
        :param input_op: input tensor
        :param categories_count: int, number of filters output of prediction head should have - basically
        we want prediction head to predict a one-hot encoding for each default fox
        :param head_configuration: dictionary with head configuration options
        :return: tensor op
        """

        # Compute total number of default boxes prediction head should make prediction for.
        # For each pixel prediction head receives it should make predictions for
        # base bounding boxes sizes count * 2 * aspect ratios count default boxes centered on that pixel.
        # We multiple aspect rations by 2, since we compute them both in horizontal and vertical orientation
        default_boxes_count = \
            len(head_configuration["base_bounding_box_sizes"]) * \
            2 * len(head_configuration["aspect_ratios"])

        # For each default box we want to make one hot encoded prediction across categories
        total_filters_count = default_boxes_count * categories_count

        x = tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='elu')(input_op)
        x = tf.keras.layers.Conv2D(filters=total_filters_count, kernel_size=(3, 3), padding='same')(x)

        # Reshape prediction to 3D matrix (batch_dimension, default boxes on all pixel locations, categories count)
        return tf.reshape(x, shape=(tf.shape(x)[0], -1, categories_count))


class VGGishModel:
    """
    Class that wraps VGGish network to provide training and prediction methods
    """

    def __init__(self, session, network):
        """
        Constructor
        :param session: tensorflow.Session instance
        :param network: SSD model network instance
        """

        self.session = session
        self.network = network
        self.learning_rate = None

        self.default_boxes_categories_ids_vector_placeholder = tf.placeholder(dtype=tf.int32, shape=(None,))

        self.loss_op = self._get_loss_op(
            default_boxes_categories_ids_vector_placeholder=self.default_boxes_categories_ids_vector_placeholder,
            batch_of_predictions_logits_matrices_op=self.network.batch_of_predictions_logits_matrices_op
        )

        self.learning_rate_placeholder = tf.placeholder(shape=None, dtype=tf.float32)
        self.train_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate_placeholder).minimize(self.loss_op)

    def train(self, data_bunch, configuration, callbacks):
        """
        Method for training network
        :param data_bunch: net.data.DataBunch instance created with SSD input data loaders
        :param configuration: dictionary with training options
        :param callbacks: list of net.Callback instances. Used to save weights, control learning rate, etc.
        """

        self.learning_rate = configuration["learning_rate"]
        epoch_index = 0

        training_data_generator = iter(data_bunch.training_data_loader)
        validation_data_generator = iter(data_bunch.validation_data_loader)

        for callback in callbacks:
            callback.model = self

        try:

            while epoch_index < configuration["epochs"]:

                print("Epoch {}/{}".format(epoch_index, configuration["epochs"]))

                epoch_log = {
                    "epoch_index": epoch_index,
                    "training_loss": self._train_for_one_epoch(
                        data_generator=training_data_generator,
                        samples_count=len(data_bunch.training_data_loader)),
                    "validation_loss": self._validate_for_one_epoch(
                        data_generator=validation_data_generator,
                        samples_count=len(data_bunch.validation_data_loader)
                    )
                }

                print(epoch_log)

                for callback in callbacks:
                    callback.on_epoch_end(epoch_log)

                epoch_index += 1

        finally:

            # Stop data generators, since they are running on a separate thread
            data_bunch.training_data_loader.stop_generator()
            data_bunch.validation_data_loader.stop_generator()

    def _train_for_one_epoch(self, data_generator, samples_count):

        losses = []

        for _ in tqdm.tqdm(range(samples_count)):

            image, default_boxes_categories_ids_vector = next(data_generator)

            feed_dictionary = {
                self.network.input_placeholder: np.array([image]),
                self.default_boxes_categories_ids_vector_placeholder: default_boxes_categories_ids_vector,
                self.learning_rate_placeholder: self.learning_rate
            }

            loss, _ = self.session.run(
                [self.loss_op, self.train_op], feed_dictionary)

            losses.append(loss)

        return np.mean(losses)

    def _validate_for_one_epoch(self, data_generator, samples_count):

        losses = []

        for _ in tqdm.tqdm(range(samples_count)):

            image, default_boxes_categories_ids_vector = next(data_generator)

            feed_dictionary = {
                self.network.input_placeholder: np.array([image]),
                self.default_boxes_categories_ids_vector_placeholder: default_boxes_categories_ids_vector
            }

            loss = self.session.run(self.loss_op, feed_dictionary)

            losses.append(loss)

        return np.mean(losses)

    @staticmethod
    def _get_loss_op(
            default_boxes_categories_ids_vector_placeholder,
            batch_of_predictions_logits_matrices_op):

        # First flatten out batch dimension for batch_of_predictions_logits_matrices_op
        # Its batch dimension should be 1, we would tensorflow to raise an exception of it isn't

        batch_of_predictions_logits_matrices_shape = tf.shape(batch_of_predictions_logits_matrices_op)
        default_boxes_count = batch_of_predictions_logits_matrices_shape[1]
        categories_count = batch_of_predictions_logits_matrices_shape[2]

        predictions_logits_matrix = tf.reshape(
            batch_of_predictions_logits_matrices_op,
            shape=(default_boxes_count, categories_count))

        raw_loss_op = tf.losses.sparse_softmax_cross_entropy(
            labels=default_boxes_categories_ids_vector_placeholder, logits=predictions_logits_matrix)

        all_ones_op = tf.ones(shape=(default_boxes_count,), dtype=tf.float32)
        all_zeros_op = tf.zeros(shape=(default_boxes_count,), dtype=tf.float32)

        # Get a selector for all positive losses, split positive losses and negatives losses
        positive_losses_selector_op = tf.where(
            default_boxes_categories_ids_vector_placeholder > 0, all_ones_op, all_zeros_op)

        positive_matches_count = tf.cast(tf.reduce_sum(positive_losses_selector_op), tf.int32)

        # Get positive losses op - that is op with losses only for default bounding boxes
        # that were matched with ground truth annotations.
        # First multiply raw losses with selector op, so that all negative losses will be zero.
        # Then sort losses in descending order and select positive_matches_count elements.
        # Thus end effect is that we select positive losses only
        positive_losses_op = tf.sort(
            raw_loss_op * positive_losses_selector_op, direction='DESCENDING')[:positive_matches_count]

        # Get negative losses op that is op with losses for default boxes that weren't matched with any ground truth
        # annotations, or should predict background, in a similar manner as we did for positive losses.
        # Choose x times positive matches count largest losses only for hard negatives mining
        negative_losses_op = tf.sort(
            raw_loss_op * (1.0 - positive_losses_selector_op), direction='DESCENDING')[:(1000 * positive_matches_count)]

        # If there were any positive matches at all, then return mean of both losses.
        # Otherwise return 0 - as we can't have a mean of an empty op.
        return tf.cond(
            pred=positive_matches_count > 0,
            true_fn=lambda: tf.reduce_mean(tf.concat([positive_losses_op, negative_losses_op], axis=0)),
            false_fn=lambda: tf.constant(0, dtype=tf.float32))

    def save(self, save_path):
        """
        Save model's network
        :param save_path: prefix for filenames created for the checkpoint
        """

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        tf.train.Saver().save(self.session, save_path)

    def load(self, save_path):
        """
        Save model's network
        :param save_path: prefix for filenames created for the checkpoint
        """

        tf.train.Saver().restore(self.session, save_path)

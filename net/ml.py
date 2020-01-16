"""
Module with machine learning code
"""

import os
import pprint

import numpy as np
import tensorflow as tf
import tqdm

import net.ssd


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

        ops_map = {
            "block2_head": vgg.get_layer("block2_pool").output,
            "block3_head": vgg.get_layer("block3_pool").output,
            "block4_head": vgg.get_layer("block4_pool").output,
            "block5_head": vgg.get_layer("block5_pool").output,
        }

        categories_predictions_heads_ops_list = []
        offset_predictions_heads_ops_list = []

        for block in model_configuration["prediction_heads_order"]:

            categories_predictions_head, offsets_predictions_head = self.get_predictions_head(
                input_op=ops_map[block],
                categories_count=categories_count,
                head_configuration=model_configuration[block])

            categories_predictions_heads_ops_list.append(categories_predictions_head)
            offset_predictions_heads_ops_list.append(offsets_predictions_head)

        self.batch_of_categories_predictions_logits_matrices_op = \
            tf.concat(categories_predictions_heads_ops_list, axis=1)

        self.batch_of_softmax_categories_predictions_matrices_op = tf.nn.softmax(
            self.batch_of_categories_predictions_logits_matrices_op, axis=-1)

        self.batch_of_offsets_predictions_matrices_op = tf.concat(offset_predictions_heads_ops_list, axis=1)

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
        self.should_continue_training = None
        self.learning_rate = None

        self.ops_map = {
            "default_boxes_categories_ids_vector_placeholder": tf.placeholder(
                dtype=tf.int32, shape=(None,), name="boxes_categories_placeholder"),
            "learning_rate_placeholder": tf.placeholder(
                shape=None, dtype=tf.float32, name="learning_rate_placeholder"),
            "batch_of_categories_predictions_logits_matrices_op":
                self.network.batch_of_categories_predictions_logits_matrices_op,
            "default_boxes_sizes_op": tf.placeholder(
                shape=(None, 2), dtype=tf.int32, name="boxes_sizes_placeholder"),
            "ground_truth_offsets_matrix_op": tf.placeholder(
                shape=(None, 4), dtype=tf.float32, name="ground_truth_offsets_placeholder"),
            "batch_of_offsets_predictions_matrices_op": self.network.batch_of_offsets_predictions_matrices_op
        }

        losses_ops = self._get_losses_ops(ops_map=self.ops_map)

        self.ops_map["loss_op"] = losses_ops[0]
        self.ops_map["categorical_loss_op"] = losses_ops[1]
        self.ops_map["offsets_loss_op"] = losses_ops[2]

        self.ops_map["train_op"] = tf.train.AdamOptimizer(
            learning_rate=self.ops_map["learning_rate_placeholder"]).minimize(self.ops_map["loss_op"])

    def train(self, data_bunch, configuration, callbacks):
        """
        Method for training network
        :param data_bunch: net.data.DataBunch instance created with SSD input data loaders
        :param configuration: dictionary with training options
        :param callbacks: list of net.Callback instances. Used to save weights, control learning rate, etc.
        """

        self.learning_rate = configuration["learning_rate"]
        self.should_continue_training = True
        epoch_index = 0

        training_data_generator = iter(data_bunch.training_data_loader)
        validation_data_generator = iter(data_bunch.validation_data_loader)

        for callback in callbacks:
            callback.model = self

        try:

            while epoch_index < configuration["epochs"] and self.should_continue_training is True:

                print("Epoch {}/{}".format(epoch_index, configuration["epochs"]))

                epoch_log = {
                    "epoch_index": epoch_index,
                    "training_losses_map": self._train_for_one_epoch(
                        data_generator=training_data_generator,
                        samples_count=len(data_bunch.training_data_loader)),
                    "validation_losses_map": self._validate_for_one_epoch(
                        data_generator=validation_data_generator,
                        samples_count=len(data_bunch.validation_data_loader))
                }

                pprint.pprint(epoch_log)

                for callback in callbacks:
                    callback.on_epoch_end(epoch_log)

                epoch_index += 1

        finally:

            # Stop data generators, since they are running on a separate thread
            data_bunch.training_data_loader.stop_generator()
            data_bunch.validation_data_loader.stop_generator()

    def _train_for_one_epoch(self, data_generator, samples_count):

        losses_map = {
            "total": [],
            "categorical": [],
            "offset": []
        }

        for _ in tqdm.tqdm(range(samples_count)):

            image, default_boxes_categories_ids_vector, default_boxes_sizes, ground_truth_offsets = \
                next(data_generator)

            feed_dictionary = {
                self.network.input_placeholder: np.array([image]),
                self.ops_map["default_boxes_categories_ids_vector_placeholder"]: default_boxes_categories_ids_vector,
                self.ops_map["learning_rate_placeholder"]: self.learning_rate,
                self.ops_map["default_boxes_sizes_op"]: default_boxes_sizes,
                self.ops_map["ground_truth_offsets_matrix_op"]: ground_truth_offsets
            }

            total_loss, categorical_loss, offsets_loss, _ = self.session.run(
                [self.ops_map["loss_op"], self.ops_map["categorical_loss_op"], self.ops_map["offsets_loss_op"],
                 self.ops_map["train_op"]], feed_dictionary)

            losses_map["total"].append(total_loss)
            losses_map["categorical"].append(categorical_loss)
            losses_map["offset"].append(offsets_loss)

        return {key: np.mean(value) for key, value in losses_map.items()}

    def _validate_for_one_epoch(self, data_generator, samples_count):

        losses_map = {
            "total": [],
            "categorical": [],
            "offset": []
        }

        for _ in tqdm.tqdm(range(samples_count)):

            image, default_boxes_categories_ids_vector, default_boxes_sizes, ground_truth_offsets = \
                next(data_generator)

            feed_dictionary = {
                self.network.input_placeholder: np.array([image]),
                self.ops_map["default_boxes_categories_ids_vector_placeholder"]: default_boxes_categories_ids_vector,
                self.ops_map["learning_rate_placeholder"]: self.learning_rate,
                self.ops_map["default_boxes_sizes_op"]: default_boxes_sizes,
                self.ops_map["ground_truth_offsets_matrix_op"]: ground_truth_offsets
            }

            total_loss, categorical_loss, offsets_loss = self.session.run(
                [self.ops_map["loss_op"], self.ops_map["categorical_loss_op"], self.ops_map["offsets_loss_op"]],
                feed_dictionary)

            losses_map["total"].append(total_loss)
            losses_map["categorical"].append(categorical_loss)
            losses_map["offset"].append(offsets_loss)

        return {key: np.mean(value) for key, value in losses_map.items()}

    @staticmethod
    def _get_losses_ops(ops_map):

        # First flatten out batch dimension for batch_of_predictions_logits_matrices_op
        # Its batch dimension should be 1, we would tensorflow to raise an exception of it isn't

        batch_of_categories_predictions_logits_matrices_shape = \
            tf.shape(ops_map["batch_of_categories_predictions_logits_matrices_op"])

        default_boxes_count = batch_of_categories_predictions_logits_matrices_shape[1]
        categories_count = batch_of_categories_predictions_logits_matrices_shape[2]

        categories_predictions_logits_matrix = tf.reshape(
            ops_map["batch_of_categories_predictions_logits_matrices_op"],
            shape=(default_boxes_count, categories_count))

        offsets_predictions_matrix_op = tf.reshape(
            ops_map["batch_of_offsets_predictions_matrices_op"],
            shape=(default_boxes_count, 4))

        losses_builder = net.ssd.SingleShotDetectorLossBuilder(
            default_boxes_categories_ids_vector_op=ops_map["default_boxes_categories_ids_vector_placeholder"],
            categories_predictions_logits_matrix_op=categories_predictions_logits_matrix,
            hard_negatives_mining_ratio=3,
            default_boxes_sizes_op=ops_map["default_boxes_sizes_op"],
            ground_truth_offsets_matrix_op=ops_map["ground_truth_offsets_matrix_op"],
            offsets_predictions_matrix_op=offsets_predictions_matrix_op)

        return losses_builder.loss_op, losses_builder.categorical_loss_op, losses_builder.offsets_loss_op

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

    def predict(self, image):
        """
        Computes prediction on a single image
        :param image: 3D numpy array representing an image
        :return: 2 elements tuple,
        (2D numpy array with softmax_predictions_matrix, 2D numpy array with offsets predictions)
        """

        feed_dictionary = {
            self.network.input_placeholder: np.array([image])
        }

        categorical_predictions_batches, offsets_predictions_batches = self.session.run(
            [self.network.batch_of_softmax_categories_predictions_matrices_op,
             self.network.batch_of_offsets_predictions_matrices_op],
            feed_dictionary)

        return categorical_predictions_batches[0], offsets_predictions_batches[0]

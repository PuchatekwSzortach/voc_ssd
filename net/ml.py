"""
Module with machine learning code
"""

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

        self.predictions_logits_matrix_op = tf.concat(predictions_heads_ops_list, axis=1)

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

    def train(self, data_bunch, configuration):
        """
        Method for training network
        :param data_bunch: net.data.DataBunch instance created with SSD input data loaders
        :param configuration: dictionary with training options
        """

        self.learning_rate = configuration["learning_rate"]
        epoch_index = 0

        training_data_generator = iter(data_bunch.training_data_loader)

        try:

            while epoch_index < configuration["epochs"]:

                print("Epoch {}/{}".format(epoch_index, configuration["epochs"]))

                epoch_log = {
                    "epoch_index": epoch_index,
                    "training_loss": self._train_for_one_epoch(
                        data_generator=training_data_generator,
                        samples_count=len(data_bunch.training_data_loader))
                }

                print(epoch_log)

                epoch_index += 1

        finally:

            # Needed once we actually start generators
            data_bunch.training_data_loader.stop_generator()
            # validation_data_generator_factory.stop_generator()

    def _train_for_one_epoch(self, data_generator, samples_count):

        # training_losses = []

        default_boxes_factory = net.ssd.DefaultBoxesFactory(self.network.model_configuration)

        # for _ in tqdm.tqdm(range(len(data_loade)):
        for _ in tqdm.tqdm(range(5)):

            image, _matched_default_boxes_indices = next(data_generator)

            default_boxes_matrix = default_boxes_factory.get_default_boxes_matrix(image.shape)

            print("Image shape: {}".format(image.shape))
            print("default_boxes_matrix shape: {}".format(default_boxes_matrix.shape))

            feed_dictionary = {self.network.input_placeholder: np.array([image])}

            predictions_logits_matrix = self.session.run(
                self.network.predictions_logits_matrix_op, feed_dictionary)

            print("predictions_logits_matrix shape: {}".format(predictions_logits_matrix.shape))

        return "fake training loss"

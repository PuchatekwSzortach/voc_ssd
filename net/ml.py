"""
Module with machine learning code
"""

import tensorflow as tf


class VGGishNetwork:
    """
    SSD model based on VGG
    """

    def __init__(self, categories_count):
        """
        Constructor
        :param categories_count: number of categories to predict, including background
        """

        vgg = tf.keras.applications.VGG16(include_top=False)

        self.input_placeholder = vgg.input

        self.ops_map = {
            "block2_pool": vgg.get_layer("block2_pool").output,
            "block3_pool": vgg.get_layer("block3_pool").output,
            "block4_pool": vgg.get_layer("block4_pool").output,
            "block5_pool": vgg.get_layer("block5_pool").output,
        }

        self.prediction_heads = {
            "block2_head": self.get_prediction_head(self.ops_map["block2_pool"], categories_count),
            "block3_head": self.get_prediction_head(self.ops_map["block3_pool"], categories_count),
            "block4_head": self.get_prediction_head(self.ops_map["block4_pool"], categories_count),
            "block5_head": self.get_prediction_head(self.ops_map["block5_pool"], categories_count)
        }

    @staticmethod
    def get_prediction_head(input_op, filters_count):
        """
        Creates a prediction head
        :param input_op: input tensor
        :param filters_count: number of filters prediction head should have
        :return: tensor op
        """

        x = tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), padding='valid', activation='elu')(input_op)
        return tf.keras.layers.Conv2D(filters=filters_count, kernel_size=(3, 3), padding='valid')(x)


class VGGishModel:
    """
    Class that wraps VGGish network to provide training and prediction methods
    """

    def __init__(self, network):
        """
        Constructor
        :param network:
        """

        self.network = network

    def train(self, training_data_generator_factory, validation_data_generator_factory, configuration, callbacks=None):
        """
        Method for training network
        :param training_data_generator_factory:
        :param validation_data_generator_factory:
        :param configuration:
        :param callbacks:
        :return:
        """
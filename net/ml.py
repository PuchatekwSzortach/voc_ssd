"""
Module with machine learning code
"""

import tensorflow as tf
import tqdm


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
        self.learning_rate = None

    def train(self, data_bunch, default_boxes_factory, configuration):
        """
        Method for training network
        :param data_bunch: net.data.DataBunch instance created with SSD input data loaders
        :param default_boxes_factory: DefaultBoxesFactory instance, creates default boxes for data from generators
        :param configuration: dictionary with training options
        """

        self.learning_rate = configuration["learning_rate"]
        epoch_index = 0

        training_data_generator = iter(data_bunch.training_data_loader)

        while epoch_index < configuration["epochs"]:

            print("Epoch {}/{}".format(epoch_index, configuration["epochs"]))

            epoch_log = {
                "epoch_index": epoch_index,
                "training_loss": self._train_for_one_epoch(
                    training_data_generator, len(data_bunch.training_data_loader))
            }

            print(epoch_log)
            epoch_index += 1

        # Needed once we actually start generators
        data_bunch.training_data_loader.stop_generator()
        # validation_data_generator_factory.stop_generator()

    def _train_for_one_epoch(self, data_generator, batches_count):

        # training_losses = []

        # for _ in tqdm.tqdm(range(batches_count)):
        # for _ in tqdm.tqdm(range(500)):
        #
        #     image, annotations = next(data_generator)

        return "fake training loss"

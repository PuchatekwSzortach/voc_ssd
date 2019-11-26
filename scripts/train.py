"""
Script to train SSD model
"""

import argparse
import sys

import tensorflow as tf
import yaml

import net.data
import net.ssd
import net.ml


def get_ssd_training_loop_data_bunch(config):
    """
    Given config, return data bunch instance with training and validation loaders
    set to net.data.BackgroundDataLoader instances that return ssd training data loaders for training and validation,
    respectively
    :param config: dictionary with configuration parameters
    :return: net.data.DataBunch instances wrapping net.data.BackgroundDataLoader instances that return
    data suitable for training ssd model
    """

    training_samples_loader = net.data.VOCSamplesDataLoader(
        data_directory=config["voc"]["data_directory"],
        data_set_path=config["voc"]["train_set_path"],
        size_factor=config["size_factor"],
        objects_filtering_config=config["objects_filtering"])

    ssd_training_samples_loader = net.ssd.SSDTrainingLoopDataLoader(
        voc_samples_data_loader=training_samples_loader,
        ssd_model_configuration=config["vggish_model_configuration"])

    validation_samples_loader = net.data.VOCSamplesDataLoader(
        data_directory=config["voc"]["data_directory"],
        data_set_path=config["voc"]["validation_set_path"],
        size_factor=config["size_factor"],
        objects_filtering_config=config["objects_filtering"])

    ssd_validation_samples_loader = net.ssd.SSDTrainingLoopDataLoader(
        voc_samples_data_loader=validation_samples_loader,
        ssd_model_configuration=config["vggish_model_configuration"])

    return net.data.DataBunch(
        training_data_loader=net.data.BackgroundDataLoader(ssd_training_samples_loader),
        validation_data_loader=net.data.BackgroundDataLoader(ssd_validation_samples_loader))


def main():
    """
    Script entry point
    """

    parser = argparse.ArgumentParser()

    parser.add_argument('--config', action="store", required=True)
    arguments = parser.parse_args(sys.argv[1:])

    with open(arguments.config) as file:
        config = yaml.safe_load(file)

    data_bunch = get_ssd_training_loop_data_bunch(config)

    network = net.ml.VGGishNetwork(
        model_configuration=config["vggish_model_configuration"],
        categories_count=len(config["categories"]))

    initialized_variables = tf.global_variables()

    session = tf.keras.backend.get_session()

    model = net.ml.VGGishModel(session, network)

    uninitialized_variables = set(tf.global_variables()).difference(initialized_variables)
    session.run(tf.variables_initializer(uninitialized_variables))

    model.train(data_bunch, config["train"])


if __name__ == "__main__":

    main()

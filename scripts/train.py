"""
Script to train SSD model
"""

import argparse
import sys

import tensorflow as tf
import yaml

import net.callbacks
import net.data
import net.ml
import net.ssd
import net.utilities


def get_ssd_training_loop_data_bunch(config, ssd_model_configuration):
    """
    Given config, return data bunch instance with training and validation loaders
    set to net.data.BackgroundDataLoader instances that return ssd training data loaders for training and validation,
    respectively
    :param config: dictionary with data paths and similar configuration parameters
    :param ssd_model_configuration: dictionary with ssd model configuration parameters
    :return: net.data.DataBunch instances wrapping net.data.BackgroundDataLoader instances that return
    data suitable for training ssd model
    """

    training_samples_loader = net.data.VOCSamplesDataLoader(
        data_directory=config["voc"]["data_directory"],
        data_set_path=config["voc"]["train_set_path"],
        categories=config["categories"],
        size_factor=config["size_factor"],
        augmentation_pipeline=net.data.get_image_augmentation_pipeline())

    ssd_training_samples_loader = net.ssd.SSDTrainingLoopDataLoader(
        voc_samples_data_loader=training_samples_loader,
        ssd_model_configuration=ssd_model_configuration)

    validation_samples_loader = net.data.VOCSamplesDataLoader(
        data_directory=config["voc"]["data_directory"],
        data_set_path=config["voc"]["validation_set_path"],
        categories=config["categories"],
        size_factor=config["size_factor"])

    ssd_validation_samples_loader = net.ssd.SSDTrainingLoopDataLoader(
        voc_samples_data_loader=validation_samples_loader,
        ssd_model_configuration=ssd_model_configuration)

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

    ssd_model_configuration = config["vggish_model_configuration"]

    network = net.ml.VGGishNetwork(
        model_configuration=ssd_model_configuration,
        categories_count=len(config["categories"]))

    initialized_variables = tf.global_variables()

    session = tf.keras.backend.get_session()

    model = net.ml.SSDModel(session, network)

    uninitialized_variables = set(tf.global_variables()).difference(initialized_variables)
    session.run(tf.variables_initializer(uninitialized_variables))

    callbacks = [
        net.callbacks.ModelCheckpoint(
            save_path=config["model_checkpoint_path"],
            skip_epochs_count=2),
        net.callbacks.EarlyStopping(
            patience=config["train"]["early_stopping_patience"]),
        net.callbacks.ReduceLearningRateOnPlateau(
            patience=config["train"]["reduce_learning_rate_patience"],
            factor=config["train"]["reduce_learning_rate_factor"]),
        net.callbacks.HistoryLogger(
            logger=net.utilities.get_logger(config["training_history_log_path"])
        )
    ]

    model.train(
        data_bunch=get_ssd_training_loop_data_bunch(
            config=config,
            ssd_model_configuration=ssd_model_configuration),
        configuration=config["train"],
        callbacks=callbacks)


if __name__ == "__main__":

    main()

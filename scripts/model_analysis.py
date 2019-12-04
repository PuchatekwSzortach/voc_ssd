"""
Script with model analysis
"""

import argparse
import sys

import tensorflow as tf
import yaml

import net.analysis
import net.data
import net.ml
import net.ssd
import net.utilities


def main():
    """
    Script entry point
    """

    parser = argparse.ArgumentParser()

    parser.add_argument('--config', action="store", required=True)
    arguments = parser.parse_args(sys.argv[1:])

    with open(arguments.config) as file:
        config = yaml.safe_load(file)

    network = net.ml.VGGishNetwork(
        model_configuration=config["vggish_model_configuration"],
        categories_count=len(config["categories"]))

    session = tf.keras.backend.get_session()

    model = net.ml.VGGishModel(session, network)
    model.load(config["best_model_checkpoint_path"])

    validation_samples_loader = net.data.VOCSamplesDataLoader(
        data_directory=config["voc"]["data_directory"],
        data_set_path=config["voc"]["validation_set_path"],
        categories=config["categories"],
        size_factor=config["size_factor"],
        objects_filtering_config=config["objects_filtering"])

    logger = net.utilities.get_logger(config["log_path"])

    default_boxes_factory = net.ssd.DefaultBoxesFactory(model_configuration=config["vggish_model_configuration"])

    net.analysis.log_precision_recall_analysis(
        logger=logger,
        model=model,
        samples_loader=validation_samples_loader,
        default_boxes_factory=default_boxes_factory,
        config=config)


if __name__ == "__main__":

    main()

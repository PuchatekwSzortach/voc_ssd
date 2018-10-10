"""
Script to train SSD model
"""

import argparse
import sys

import yaml

import net.data
import net.ssd
import net.ml


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument('--config', action="store", required=True)
    arguments = parser.parse_args(sys.argv[1:])

    with open(arguments.config) as file:
        config = yaml.safe_load(file)

    training_samples_generator_factory = net.data.VOCSamplesGeneratorFactory(
        config["voc"]["data_directory"], config["voc"]["validation_set_path"], config["size_factor"])

    training_input_generator_factory = net.data.SSDInputGeneratorFactory(
        training_samples_generator_factory, config["objects_filtering"])

    validation_samples_generator_factory = net.data.VOCSamplesGeneratorFactory(
        config["voc"]["data_directory"], config["voc"]["validation_set_path"], config["size_factor"])

    validation_input_generator_factory = net.data.SSDInputGeneratorFactory(
        validation_samples_generator_factory, config["objects_filtering"])

    default_boxes_factory = net.ssd.DefaultBoxesFactory(config["vggish_model_configuration"])

    network = net.ml.VGGishNetwork(len(config["categories"]))
    model = net.ml.VGGishModel(network)


if __name__ == "__main__":

    main()

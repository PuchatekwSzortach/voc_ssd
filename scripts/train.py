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
    """
    Script entry point
    """

    parser = argparse.ArgumentParser()

    parser.add_argument('--config', action="store", required=True)
    arguments = parser.parse_args(sys.argv[1:])

    with open(arguments.config) as file:
        config = yaml.safe_load(file)

    training_samples_loader = net.data.VOCSamplesDataLoader(
        config["voc"]["data_directory"], config["voc"]["train_set_path"], config["size_factor"])

    training_input_data_loader = net.data.SSDModelInputDataLoader(training_samples_loader, config["objects_filtering"])

    validation_samples_loader = net.data.VOCSamplesDataLoader(
        config["voc"]["data_directory"], config["voc"]["validation_set_path"], config["size_factor"])

    validation_input_data_loader = net.data.SSDModelInputDataLoader(
        validation_samples_loader, config["objects_filtering"])

    data_bunch = net.data.DataBunch(
        training_data_loader=training_input_data_loader,
        validation_data_loader=validation_input_data_loader)

    default_boxes_factory = net.ssd.DefaultBoxesFactory(config["vggish_model_configuration"])

    network = net.ml.VGGishNetwork(len(config["categories"]))
    model = net.ml.VGGishModel(network)

    model.train(data_bunch, default_boxes_factory, config["train"])


if __name__ == "__main__":

    main()

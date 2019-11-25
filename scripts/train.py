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
        data_directory=config["voc"]["data_directory"],
        data_set_path=config["voc"]["train_set_path"],
        size_factor=config["size_factor"],
        objects_filtering_config=config["objects_filtering"])

    validation_samples_loader = net.data.VOCSamplesDataLoader(
        data_directory=config["voc"]["data_directory"],
        data_set_path=config["voc"]["validation_set_path"],
        size_factor=config["size_factor"],
        objects_filtering_config=config["objects_filtering"])

    data_bunch = net.data.DataBunch(
        training_data_loader=training_samples_loader,
        validation_data_loader=validation_samples_loader)

    default_boxes_factory = net.ssd.DefaultBoxesFactory(config["vggish_model_configuration"])

    network = net.ml.VGGishNetwork(len(config["categories"]))
    model = net.ml.VGGishModel(network)

    model.train(data_bunch, default_boxes_factory, config["train"])


if __name__ == "__main__":

    main()

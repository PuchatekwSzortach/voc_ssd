"""
Script with visualizations of data generators outputs, model prediction, etc
"""

import argparse
import sys

import yaml
import tqdm
import vlogging

import net.utilities
import net.data


def log_voc_samples_generator_output(logger, config):
    """
    Logs voc samples generator output
    """

    generator = net.data.VOCSamplesGeneratorFactory(
        config["voc"]["data_directory"], config["voc"]["validation_set_path"]).get_generator()

    for _ in tqdm.tqdm(range(5)):

        image, _ = next(generator)
        logger.info(vlogging.VisualRecord("Data", [image]))


def main():
    """
    Script entry point
    """

    parser = argparse.ArgumentParser()

    parser.add_argument('--config', action="store", required=True)
    arguments = parser.parse_args(sys.argv[1:])

    with open(arguments.config) as file:
        config = yaml.safe_load(file)

    logger = net.utilities.get_logger(config["log_path"])

    log_voc_samples_generator_output(logger, config)


if __name__ == "__main__":

    main()

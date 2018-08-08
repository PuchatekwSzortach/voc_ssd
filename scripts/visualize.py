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
        config["voc"]["data_directory"], config["voc"]["validation_set_path"], config["size_factor"]).get_generator()

    categories_to_colors_map = net.utilities.get_categories_to_colors_map(config["categories"])

    for _ in tqdm.tqdm(range(10)):

        image, annotations = next(generator)

        colors = [categories_to_colors_map[annotation.label] for annotation in annotations]

        image = net.utilities.get_annotated_image(
            image, annotations, colors, config["font_path"])

        labels = [annotation.label for annotation in annotations]
        message = "{} - {}".format(image.shape[:2], labels)

        logger.info(vlogging.VisualRecord("Data", [image], message))


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

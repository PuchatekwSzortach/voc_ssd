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

    for _ in tqdm.tqdm(range(100)):

        image, annotations = next(generator)

        colors = [categories_to_colors_map[annotation.label] for annotation in annotations]

        image = net.utilities.get_annotated_image(
            image, annotations, colors, config["font_path"])

        labels = [annotation.label for annotation in annotations]
        message = "{} - {}".format(image.shape[:2], labels)

        logger.info(vlogging.VisualRecord("Data", [image], message))


def log_sample_with_odd_sized_annotation(logger, image, annotations, categories_to_colors_map, font_path):
    """
    Logs into a logger a single image and its annotations, writing dimensions details of each annotation
    :param logger: logger instance
    :param image: image to log
    :param annotations: annotations for the image to log
    :param categories_to_colors_map: dictionary mapping categories labels to colors
    :param font_path: path to font to be used to draw labels text on image
    """

    colors = [categories_to_colors_map[annotation.label] for annotation in annotations]

    image = net.utilities.get_annotated_image(image, annotations, colors, font_path)

    labels = []

    for annotation in annotations:
        label = "{} - width: {}, height: {}, aspect ratio: {}".format(
            annotation.label, annotation.width, annotation.height, annotation.aspect_ratio)

        labels.append(label)

    labels_message = "\n".join(labels)
    message = "{}\n{}".format(image.shape[:2], labels_message)

    logger.info(vlogging.VisualRecord("Data", [image], message))


def log_samples_with_odd_sized_annotations(logger, config):
    """
    Logs images with objects of odd sizes - suspiciously small, strange aspect ratios, etc
    """

    generator_factory = net.data.VOCSamplesGeneratorFactory(
        config["voc"]["data_directory"], config["voc"]["validation_set_path"], config["size_factor"])

    generator = generator_factory.get_generator()

    categories_to_colors_map = net.utilities.get_categories_to_colors_map(config["categories"])

    all_annotations_count = 0
    unusual_sized_annotation_count = 0

    for _ in tqdm.tqdm(range(generator_factory.get_size())):

        image, annotations = next(generator)

        all_annotations_count += len(annotations)

        unusual_sized_annotations = []

        for annotation in annotations:

            if net.utilities.is_annotation_size_unusual(
                    annotation,
                    config["objects_filtering"]["minimum_size"],
                    config["objects_filtering"]["minimum_aspect_ratio"],
                    config["objects_filtering"]["maximum_aspect_ratio"]):

                unusual_sized_annotations.append(annotation)

        unusual_sized_annotation_count += len(unusual_sized_annotations)

        if len(unusual_sized_annotations) > 0:

            log_sample_with_odd_sized_annotation(
                logger, image, unusual_sized_annotations, categories_to_colors_map, config["font_path"])

    print("Unusual object sizes counts to objects count: {}/{}".format(
        unusual_sized_annotation_count, all_annotations_count))


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
    # log_samples_with_odd_sized_annotations(logger, config)


if __name__ == "__main__":

    main()

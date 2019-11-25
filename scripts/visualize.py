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
import net.ssd
import net.plot


def log_voc_samples_generator_output(logger, config):
    """
    Logs voc samples generator output
    """

    samples_loader = net.data.VOCSamplesDataLoader(
        config["voc"]["data_directory"], config["voc"]["validation_set_path"], config["size_factor"])

    generator = iter(samples_loader)

    categories_to_colors_map = net.utilities.get_categories_to_colors_map(config["categories"])

    for _ in tqdm.tqdm(range(100)):

        image, annotations = next(generator)

        colors = [categories_to_colors_map[annotation.label] for annotation in annotations]

        image = net.plot.get_annotated_image(image, annotations, colors, config["font_path"])

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

    image = net.plot.get_annotated_image(image, annotations, colors, font_path)

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

    samples_loader = net.data.VOCSamplesDataLoader(
        config["voc"]["data_directory"], config["voc"]["validation_set_path"], config["size_factor"])

    generator = iter(samples_loader)

    categories_to_colors_map = net.utilities.get_categories_to_colors_map(config["categories"])

    all_annotations_count = 0
    unusual_sized_annotation_count = 0

    for _ in tqdm.tqdm(range(len(samples_loader))):

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


def log_default_boxes_matches_for_single_sample(
        logger, ssd_input_generator, default_boxes_factory, categories_to_colors_map, font_path):
    """
    Logs default boxes matches for a single sample. If no default box was matched with any annotation,
    nothing is logger.
    :param logger: logger instance
    :param ssd_input_generator: generator that outputs (image, annotations) tuples
    :param default_boxes_factory: net.ssd.DefaultBoxesFactory instance
    :param categories_to_colors_map: dictionary mapping categories labels to colors
    :param font_path: path to font to be used to annotate images
    """

    image, annotations = next(ssd_input_generator)
    default_boxes_matrix = default_boxes_factory.get_default_boxes_matrix(image.shape)

    all_matched_default_boxes_indices = []

    for annotation in annotations:
        matched_default_boxes_indices = net.utilities.get_matched_boxes_indices(
            annotation.bounding_box, default_boxes_matrix)

        all_matched_default_boxes_indices.extend(matched_default_boxes_indices.tolist())

    if len(all_matched_default_boxes_indices) > 0:
        annotations_colors = [categories_to_colors_map[annotation.label] for annotation in annotations]

        annotated_image = net.plot.get_annotated_image(
            image, annotations, colors=annotations_colors, draw_labels=True, font_path=font_path)

        matched_boxes = default_boxes_matrix[all_matched_default_boxes_indices]
        matched_boxes_image = net.plot.get_image_with_boxes(image, matched_boxes, color=(0, 255, 0))

        logger.info(vlogging.VisualRecord("Default boxes", [annotated_image, matched_boxes_image]))


def log_default_boxes_matches(logger, config):
    """
    Log default boxes matches
    """

    samples_loader = net.data.VOCSamplesDataLoader(
        config["voc"]["data_directory"], config["voc"]["validation_set_path"], config["size_factor"])

    ssd_input_data_loader = net.data.SSDModelInputDataLoader(samples_loader, config["objects_filtering"])

    default_boxes_factory = net.ssd.DefaultBoxesFactory(config["vggish_model_configuration"])
    categories_to_colors_map = net.utilities.get_categories_to_colors_map(config["categories"])

    for _ in tqdm.tqdm(range(100)):

        log_default_boxes_matches_for_single_sample(
            logger, iter(ssd_input_data_loader), default_boxes_factory, categories_to_colors_map, config["font_path"])

    ssd_input_data_loader.stop_generator()


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

    # log_voc_samples_generator_output(logger, config)
    # log_samples_with_odd_sized_annotations(logger, config)

    log_default_boxes_matches(logger, config)


if __name__ == "__main__":

    main()

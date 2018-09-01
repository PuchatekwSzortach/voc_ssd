"""
Script analyzing bounding boxes sizes and aspect ratios for VOC dataset
"""

import argparse
import sys
import os
import collections

import yaml
import tqdm
import xmltodict

import net.data
import net.utilities


def analyze_images_sizes(config):
    """
    Analyze image sizes
    :param config: configuration dictionary
    """

    images_filenames = net.data.get_dataset_filenames(
        config["voc"]["data_directory"], config["voc"]["train_and_validation_set_path"])

    sizes = []

    for image_filename in tqdm.tqdm(images_filenames):

        annotations_path = os.path.join(config["voc"]["data_directory"], "Annotations", image_filename + ".xml")

        with open(annotations_path) as file:

            annotations = xmltodict.parse(file.read())

            size = int(annotations["annotation"]["size"]["height"]), int(annotations["annotation"]["size"]["width"])
            adjusted_size = net.utilities.get_target_shape(size, config["size_factor"])

            sizes.append(adjusted_size)

    sizes_counter = collections.Counter(sizes)
    ordered_sizes = sorted(sizes_counter.items(), key=lambda x: x[1], reverse=True)

    for size, count in ordered_sizes:
        print("{} -> {}".format(count, size))


def get_filtered_dataset_annotations(config):
    """
    Retrieves annotations for the dataset, scales them in accordance to how their images would be scaled
    in prediction, filters ount unusually sized annotations, then returns annotations that made it through filtering
    :param config: configuration dictionary
    :return: list of net.utilities.Annotation instances
    """

    images_filenames = net.data.get_dataset_filenames(
        config["voc"]["data_directory"], config["voc"]["validation_set_path"])

    annotations_paths = [os.path.join(config["voc"]["data_directory"], "Annotations", image_filename + ".xml")
                         for image_filename in images_filenames]

    all_annotations = []

    for annotations_path in tqdm.tqdm(annotations_paths):

        with open(annotations_path) as file:

            image_annotations_xml = xmltodict.parse(file.read())

            image_size = \
                int(image_annotations_xml["annotation"]["size"]["height"]), \
                int(image_annotations_xml["annotation"]["size"]["width"])

            # Read annotations
            annotations = net.data.get_objects_annotations(image_annotations_xml)

            # Resize annotations in line with how we would resize the image
            annotations = [annotation.resize(image_size, config["size_factor"]) for annotation in annotations]

            # Discard odd sized annotations
            annotations = \
                [annotation for annotation in annotations
                 if not net.utilities.is_annotation_size_unusual(annotation, **config["objects_filtering"])]

            all_annotations.extend(annotations)

    return all_annotations


def main():
    """
    Script entry point
    """

    parser = argparse.ArgumentParser()

    parser.add_argument('--config', action="store", required=True)
    arguments = parser.parse_args(sys.argv[1:])

    with open(arguments.config) as file:
        config = yaml.safe_load(file)

    # analyze_images_sizes(config)

    annotations = get_filtered_dataset_annotations(config)
    net.utilities.analyze_annotations(annotations)


if __name__ == "__main__":

    main()

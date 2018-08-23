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


def analyze_objects_sizes(config):
    """
    Analyze objects sizes
    :param config: configuration dictionary
    """

    images_filenames = net.data.get_dataset_filenames(
        config["voc"]["data_directory"], config["voc"]["train_and_validation_set_path"])

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

    sizes = [annotation.size for annotation in all_annotations]

    # Within a small margin, force objects to be the same size, so we can see frequent sizes groups more easily
    sizes = [net.utilities.get_target_shape(size, size_factor=5) for size in sizes]

    sizes_counter = collections.Counter(sizes)
    ordered_sizes = sorted(sizes_counter.items(), key=lambda x: x[1], reverse=True)

    for size, count in ordered_sizes[:500]:
        print("{} -> {}".format(count, size))


def analyze_objects_aspect_ratios(config):
    """
    Analyze objects aspect ratios
    :param config: configuration dictionary
    """

    images_filenames = net.data.get_dataset_filenames(
        config["voc"]["data_directory"], config["voc"]["train_and_validation_set_path"])

    annotations_paths = [os.path.join(config["voc"]["data_directory"], "Annotations", image_filename + ".xml")
                         for image_filename in images_filenames]

    objects_sizes = net.data.get_resized_dataset_objects_sizes(annotations_paths, config["size_factor"], verbose=True)

    # Within a small margin, force objects to be the same size, so we can see frequent sizes groups more easily
    adjusted_object_sizes = [net.utilities.get_target_shape(size, size_factor=5) for size in objects_sizes]

    aspect_ratios = [width / height for (height, width) in adjusted_object_sizes]
    print(*aspect_ratios, sep="\n")
    print(len(aspect_ratios))


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
    analyze_objects_sizes(config)
    # analyze_objects_aspect_ratios(config)


if __name__ == "__main__":

    main()

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


def get_adjusted_object_sizes(image_annotations, size_factor):
    """
    Given image annotations dictionary and size factor,
    return list of object sizes adjusted to a multiple of size factor
    :param image_annotations: dictionary with image annotations
    :param size_factor: integer, factor a multiple of which sizes should be adjusted to
    :return: list of (object height, object width) tuples
    """

    objects_annotations = net.data.get_objects_annotations(image_annotations)

    image_size = \
        int(image_annotations["annotation"]["size"]["height"]), \
        int(image_annotations["annotation"]["size"]["width"])

    target_shape = net.utilities.get_target_shape(image_size, size_factor)

    y_resize_fraction = target_shape[0] / image_size[0]
    x_resize_fraction = target_shape[1] / image_size[1]

    objects_sizes = []

    for object_annotation in objects_annotations:

        x_min, y_min, x_max, y_max = object_annotation.bounding_box

        resized_bounding_box = \
            round(x_min * x_resize_fraction), round(y_min * y_resize_fraction), \
            round(x_max * x_resize_fraction), round(y_max * y_resize_fraction)

        object_size = \
            resized_bounding_box[3] - resized_bounding_box[1], \
            resized_bounding_box[2] - resized_bounding_box[0]

        objects_sizes.append(object_size)

    return objects_sizes


def analyze_objects_sizes(config):
    """
    Analyze objects sizes
    :param config: configuration dictionary
    """

    images_filenames = net.data.get_dataset_filenames(
        config["voc"]["data_directory"], config["voc"]["train_and_validation_set_path"])

    objects_sizes = []

    for image_filename in tqdm.tqdm(images_filenames):

        annotations_path = os.path.join(config["voc"]["data_directory"], "Annotations", image_filename + ".xml")

        with open(annotations_path) as file:

            image_annotations = xmltodict.parse(file.read())

            image_objects_sizes = get_adjusted_object_sizes(image_annotations, config["size_factor"])
            objects_sizes.extend(image_objects_sizes)

    adjusted_object_sizes = [net.utilities.get_target_shape(size, size_factor=5) for size in objects_sizes]

    sizes_counter = collections.Counter(adjusted_object_sizes)
    ordered_sizes = sorted(sizes_counter.items(), key=lambda x: x[1], reverse=True)

    for size, count in ordered_sizes[:500]:
        print("{} -> {}".format(count, size))


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


if __name__ == "__main__":

    main()

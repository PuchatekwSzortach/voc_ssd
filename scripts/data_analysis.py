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

    objects_sizes = net.data.get_adjusted_dataset_objects_sizes(annotations_paths, config["size_factor"], verbose=True)

    for object_size in objects_sizes:

        adjusted_object_size = net.utilities.get_target_shape(object_size, size_factor=5)

        if adjusted_object_size[0] == 0 or adjusted_object_size[1] == 0:
            raise ValueError("Object size {} after adjusting is {}".format(object_size, adjusted_object_size))

    # # Within a small margin, force objects to be the same size, so we can see frequent sizes groups more easily
    # adjusted_object_sizes = [net.utilities.get_target_shape(size, size_factor=5) for size in objects_sizes]
    #
    # for object_size in adjusted_object_sizes:
    #
    #     if object_size[0] == 0 or object_size[1] == 0:
    #         raise ValueError("Object size is {}".format(object_size))

    # sizes_counter = collections.Counter(adjusted_object_sizes)
    # ordered_sizes = sorted(sizes_counter.items(), key=lambda x: x[1], reverse=True)
    #
    # for size, count in ordered_sizes[:500]:
    #     print("{} -> {}".format(count, size))


def analyze_objects_aspect_ratios(config):
    """
    Analyze objects aspect ratios
    :param config: configuration dictionary
    """

    images_filenames = net.data.get_dataset_filenames(
        config["voc"]["data_directory"], config["voc"]["train_and_validation_set_path"])

    annotations_paths = [os.path.join(config["voc"]["data_directory"], "Annotations", image_filename + ".xml")
                         for image_filename in images_filenames]

    objects_sizes = net.data.get_adjusted_dataset_objects_sizes(annotations_paths, config["size_factor"], verbose=True)

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

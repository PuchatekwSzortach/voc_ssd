"""
Data generators and other data-related code
"""

import os
import copy
import random

import cv2


def get_dataset_filenames(data_directory, data_set_path):
    """
    Get a list of filenames for the dataset
    :param data_directory: path to data directory
    :param data_set_path: path to file containing dataset filenames. This path is relative to data_directory
    :return: list of strings, filenames of images used in dataset
    """

    with open(os.path.join(data_directory, data_set_path)) as file:

        return [line.strip() for line in file.readlines()]


class VOCSamplesGeneratorFactory:
    """
    Factory class creating data batches generators that yield (image, bounding boxes) pairs
    """

    def __init__(self, data_directory, data_set_path):
        """
        Constructor
        :param data_directory: path to VOC dataset directory
        :param data_set_path: path to file listing images to be used - for selecting between train and validation
        data sets
        """

        self.data_directory = data_directory
        self.images_filenames = get_dataset_filenames(data_directory, data_set_path)

    def get_generator(self):
        """
        Returns generator that yields samples (image, bounding boxes)
        :return: generator
        """

        local_images_filenames = copy.deepcopy(self.images_filenames)

        while True:

            random.shuffle(local_images_filenames)

            for image_filename in local_images_filenames:

                image_path = os.path.join(self.data_directory, "JPEGImages", image_filename + ".jpg")
                image = cv2.imread(image_path)

                # annotations_path = os.path.join(self.data_directory, "Annotations", image_filename + ".xml")
                # print(annotations_path)
                bounding_boxes = [[100, 100, 200, 200]]

                yield image, bounding_boxes

    def get_size(self):
        """
        Gets size of dataset served by the generator
        :return: int
        """
        return len(self.images_filenames)

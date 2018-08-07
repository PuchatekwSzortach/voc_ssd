"""
Data generators and other data-related code
"""

import os
import copy
import random

import xmltodict
import cv2

import net.utilities


def get_dataset_filenames(data_directory, data_set_path):
    """
    Get a list of filenames for the dataset
    :param data_directory: path to data directory
    :param data_set_path: path to file containing dataset filenames. This path is relative to data_directory
    :return: list of strings, filenames of images used in dataset
    """

    with open(os.path.join(data_directory, data_set_path)) as file:

        return [line.strip() for line in file.readlines()]


def get_annotations(annotations_path):
    """
    Given path to image annotations, return a list of annotations for that image
    :param annotations_path: path to annotations xml
    :return: list of net.utility.Annotation objects
    """

    with open(annotations_path) as file:

        annotations = xmltodict.parse(file.read())
        xml_annotations = annotations["annotation"]["object"]

    # If image contains only a single object, annotations["annotation"]["object"] returns
    # a single OrderedDictionary. For multiple objects it returns a list of OrderedDictonaries.
    # We will wrap a single object into a list with a single element for uniform treatment
    if not isinstance(xml_annotations, list):
        xml_annotations = [xml_annotations]

    annotations = []

    for xml_annotation in xml_annotations:

        bounding_box = [
            int(xml_annotation["bndbox"]["xmin"]), int(xml_annotation["bndbox"]["ymin"]),
            int(xml_annotation["bndbox"]["xmax"]), int(xml_annotation["bndbox"]["ymax"])]

        annotation = net.utilities.Annotation(bounding_box=bounding_box, category=xml_annotation["name"])
        annotations.append(annotation)

    return annotations


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
        Returns generator that yields samples (image, annotations)
        :return: generator
        """

        local_images_filenames = copy.deepcopy(self.images_filenames)

        while True:

            random.shuffle(local_images_filenames)

            for image_filename in local_images_filenames:

                image_path = os.path.join(self.data_directory, "JPEGImages", image_filename + ".jpg")
                image = cv2.imread(image_path)

                annotations_path = os.path.join(self.data_directory, "Annotations", image_filename + ".xml")
                annotations = get_annotations(annotations_path)

                yield image, annotations

    def get_size(self):
        """
        Gets size of dataset served by the generator
        :return: int
        """
        return len(self.images_filenames)

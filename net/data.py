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


def get_objects_annotations(image_annotations):
    """
    Given an image annotations object, return a list of objects annotations
    :param image_annotations: dictionary with image annotations
    :return: list of net.utility.Annotation objects
    """

    # raw_objects_annotations is a dictionary or a list of dictionaries
    raw_objects_annotations = image_annotations["annotation"]["object"]

    # If image contains only a single object, raw_objects_annotations["annotation"]["object"] returns
    # a single OrderedDictionary. For multiple objects it returns a list of OrderedDictonaries.
    # We will wrap a single object into a list with a single element for uniform treatment
    if not isinstance(raw_objects_annotations, list):
        raw_objects_annotations = [raw_objects_annotations]

    annotations = []

    for raw_object_annotation in raw_objects_annotations:

        bounding_box = [
            int(raw_object_annotation["bndbox"]["xmin"]), int(raw_object_annotation["bndbox"]["ymin"]),
            int(raw_object_annotation["bndbox"]["xmax"]), int(raw_object_annotation["bndbox"]["ymax"])]

        annotation = net.utilities.Annotation(bounding_box=bounding_box, label=raw_object_annotation["name"])
        annotations.append(annotation)

    return annotations


class VOCSamplesGeneratorFactory:
    """
    Factory class creating data batches generators that yield (image, bounding boxes) pairs
    """

    def __init__(self, data_directory, data_set_path, size_factor):
        """
        Constructor
        :param data_directory: path to VOC dataset directory
        :param data_set_path: path to file listing images to be used - for selecting between train and validation
        :param size_factor: size factor to which images should be rescaled
        data sets
        """

        self.data_directory = data_directory
        self.images_filenames = get_dataset_filenames(data_directory, data_set_path)
        self.size_factor = size_factor

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

                with open(annotations_path) as file:

                    image_annotations = xmltodict.parse(file.read())

                objects_annotations = get_objects_annotations(image_annotations)

                bounding_boxes = [annotation.bounding_box for annotation in objects_annotations]

                image, resized_bounding_boxes = net.utilities.get_resized_sample(
                    image, bounding_boxes, size_factor=self.size_factor)

                for index, bounding_box in enumerate(resized_bounding_boxes):
                    objects_annotations[index].bounding_box = bounding_box

                yield image, objects_annotations

    def get_size(self):
        """
        Gets size of dataset served by the generator
        :return: int
        """
        return len(self.images_filenames)

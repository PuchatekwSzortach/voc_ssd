"""
Data generators and other data-related code
"""

import copy
import os
import random
import queue
import threading

import imgaug.augmenters
import numpy as np
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


def get_objects_annotations(image_annotations, labels_to_categories_index_map):
    """
    Given an image annotations object, return a list of objects annotations
    :param image_annotations: dictionary with image annotations
    :param labels_to_categories_index_map: dictionary {str: int}, specifies numerical index for each category label
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

        annotation = net.utilities.Annotation(
            bounding_box=bounding_box,
            label=raw_object_annotation["name"],
            category_id=labels_to_categories_index_map[raw_object_annotation["name"]])

        annotations.append(annotation)

    return annotations


class ImageProcessor:
    """
    Simple class wrapping up normalization and denormalization routines
    """

    @staticmethod
    def get_normalized_image(image):
        """
        Get normalized image
        :param image: numpy array
        :return: numpy array
        """

        return np.float32(image / 255.0) - 0.5

    @staticmethod
    def get_denormalized_image(image):
        """
        Transform normalized image back to original scale
        :param image: numpy array
        :return: numpy array
        """

        return np.uint8(255 * (image + 0.5))


class DataBunch:
    """
    A simple container for training and validation generators
    """

    def __init__(self, training_data_loader, validation_data_loader):

        self.training_data_loader = training_data_loader
        self.validation_data_loader = validation_data_loader


class VOCSamplesDataLoader:
    """
    Data loader that yields (image, annotations) pairs
    """

    def __init__(
            self, data_directory, data_set_path,
            categories, size_factor, augmentation_pipeline=None):
        """
        Constructor
        :param data_directory: path to VOC dataset directory
        :param data_set_path: path to file listing images to be used - for selecting between train and validation
        :param categories: list of strings, indices correspond to ids we want to give to numerical representation
        of each label
        :param size_factor: size factor to which images should be rescaled
        data sets
        will drop annotations based on filtering options
        :param augmentation_pipeline: imagaug.augmenters.Augmenter instance, optional, if not None, then it's used
        to augment image
        """

        self.data_directory = data_directory
        self.images_filenames = get_dataset_filenames(data_directory, data_set_path)

        self.labels_to_categories_index_map = {label: index for (index, label) in enumerate(categories)}

        self.size_factor = size_factor
        self.augmentation_pipeline = augmentation_pipeline

    def __len__(self):

        return len(self.images_filenames)

    def __iter__(self):

        local_images_filenames = copy.deepcopy(self.images_filenames)

        while True:

            random.shuffle(local_images_filenames)

            for image_filename in local_images_filenames:

                image_path = os.path.join(self.data_directory, "JPEGImages", image_filename + ".jpg")
                image = cv2.imread(image_path)

                annotations_path = os.path.join(self.data_directory, "Annotations", image_filename + ".xml")

                with open(annotations_path) as file:

                    image_annotations = xmltodict.parse(file.read())

                annotations = get_objects_annotations(image_annotations, self.labels_to_categories_index_map)

                if self.augmentation_pipeline is not None:

                    image, annotations = self._get_augmented_sample(image, annotations)

                bounding_boxes = [annotation.bounding_box for annotation in annotations]

                image, resized_bounding_boxes = net.utilities.get_resized_sample(
                    image, bounding_boxes, size_factor=self.size_factor)

                for index, bounding_box in enumerate(resized_bounding_boxes):
                    annotations[index].bounding_box = bounding_box

                yield ImageProcessor.get_normalized_image(image), annotations

    def _get_augmented_sample(self, image, annotations):
        """
        Augment samples
        :param image: np.array
        :param annotations: list of net.utilities.Annotation nstances
        :return: tuple (image, annotations)
        """

        bounding_boxes_container = imgaug.augmentables.BoundingBoxesOnImage(
            bounding_boxes=[imgaug.augmentables.BoundingBox(*annotation.bounding_box) for annotation in annotations],
            shape=image.shape)

        augmented_image, augmented_bounding_boxes_container = self.augmentation_pipeline(
            image=image,
            bounding_boxes=bounding_boxes_container)

        augmented_annotations = [net.utilities.Annotation(
            bounding_box=[bounding_box.x1_int, bounding_box.y1_int, bounding_box.x2_int, bounding_box.y2_int],
            label=annotation.label,
            category_id=annotation.category_id
        ) for (annotation, bounding_box) in zip(annotations, augmented_bounding_boxes_container.bounding_boxes)]

        return augmented_image, augmented_annotations


class BackgroundDataLoader:
    """
    Data loader that loads data in the background, passing it to foreground through a queue
    """

    def __init__(self, data_loader):
        """
        Constructor
        :param data_loader: data_loader instance which from which data will be loaded in the background
        """

        self.data_loader = data_loader

        self._samples_queue = queue.Queue(maxsize=100)
        self._samples_generation_thread = None
        self._continue_generating_samples = None

    def __len__(self):

        return len(self.data_loader)

    def __iter__(self):

        self._continue_generating_samples = True

        self._samples_generation_thread = threading.Thread(
            target=self._samples_generation_task,
            args=(iter(self.data_loader), self._samples_queue))

        self._samples_generation_thread.start()

        while True:

            sample = self._samples_queue.get()
            self._samples_queue.task_done()
            yield sample

    def _samples_generation_task(self, data_generator, samples_queue):

        while self._continue_generating_samples is True:

            sample = next(data_generator)
            samples_queue.put(sample)

    def stop_generator(self):
        """
        Signal data loading thread to finish working and purge the data queue.
        """

        self._continue_generating_samples = False

        while not self._samples_queue.empty():
            self._samples_queue.get()
            self._samples_queue.task_done()

        self._samples_queue.join()
        self._samples_generation_thread.join()


def get_image_augmentation_pipeline():
    """
    Get image augmentation pipeline
    :return: imgaug.augmenters.Augmenter instance
    """

    return imgaug.augmenters.Sequential(
        children=[
            imgaug.augmenters.SomeOf(
                n=(0, 3),
                children=[],
                random_order=True),
            imgaug.augmenters.Affine(scale=(0.8, 1.2)),
            # imgaug.augmenters.Affine(rotate=(-15, 15)),
            # Left-right flip
            imgaug.augmenters.Fliplr(0.5)])

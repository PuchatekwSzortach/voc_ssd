"""
Data generators and other data-related code
"""

import os
import copy
import random
import threading
import queue

import xmltodict
import tqdm
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
    Factory class creating data generator that yield (image, bounding boxes) pairs
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


class SSDInputGeneratorFactory:
    """
    Factory class creating data generator that can be used to feed an SSD model
    """

    def __init__(self, voc_samples_generator_factory, objects_filtering_config):
        """
        Constructor
        :param voc_samples_generator_factory: factory that creates generator yielding (image, annotations) tuples
        from VOC dataset
        :param objects_filtering_config: dictionary with options for objects filtering
        """

        self.voc_samples_generator_factory = voc_samples_generator_factory
        self.objects_filtering_config = objects_filtering_config

        self._samples_queue = queue.Queue(maxsize=100)
        self._samples_generation_thread = None
        self._continue_generating_samples = None

    def get_generator(self):
        """
        Returns generator that yields samples (image, annotations)
        :return: generator
        """

        self._continue_generating_samples = True

        self._samples_generation_thread = threading.Thread(
            target=self._samples_generation_task,
            args=(self.voc_samples_generator_factory, self._samples_queue, self.objects_filtering_config))

        self._samples_generation_thread.start()

        while True:

            sample = self._samples_queue.get()
            self._samples_queue.task_done()
            yield sample

    def _samples_generation_task(self, voc_samples_generator_factory, samples_queue, objects_filtering_config):

        generator = voc_samples_generator_factory.get_generator()

        while self._continue_generating_samples is True:

            image, annotations = next(generator)

            # Discard odd sized annotations
            annotations = \
                [annotation for annotation in annotations
                 if not net.utilities.is_annotation_size_unusual(annotation, **objects_filtering_config)]

            sample = image, annotations
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

    def get_size(self):
        """
        Gets size of dataset served by the generator
        :return: int
        """
        return self.voc_samples_generator_factory.get_size()


def get_resized_objects_sizes(image_annotations, size_factor):
    """
    Given image annotations dictionary and size factor,
    return list of object sizes that are resized as if the image they are contained in was resized to be
    a multiple of size factor
    :param image_annotations: dictionary with image annotations
    :param size_factor: integer, factor a multiple of which image containing annotations would be resized to,
    thus influencing how annotated objects will be resized as well
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


def get_resized_dataset_objects_sizes(annotations_paths, size_factor, verbose=False):
    """
    Given annotations paths and size factor a multiple of which we want resized images to be,
    return all objects sizes from all annotations. Objects are resized in line with hypothetical image resize
    given by size_factor
    :param annotations_paths: list of strings, paths to annotation files
    :param size_factor: integer
    :param verbose: bool, specifies whether progress bar should be displayed
    :return: list of (height, width) tuples
    """

    objects_sizes = []

    for annotations_path in tqdm.tqdm(annotations_paths, disable=not verbose):

        with open(annotations_path) as file:
            image_annotations = xmltodict.parse(file.read())

            # Since we are resizing image to be a multiple of 32 before feeding them to the network,
            # we need to do the same to objects inside the image
            image_objects_sizes = net.data.get_resized_objects_sizes(image_annotations, size_factor)
            objects_sizes.extend(image_objects_sizes)

    return objects_sizes

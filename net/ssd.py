"""
Module with SSD-specific computations
"""

import numpy as np

import net.utilities


class DefaultBoxesFactory:
    """
    Class for generating default boxes
    """

    def __init__(self, model_configuration):
        """
        Constructor
        :param model_configuration: dictionary with SSD model configuration options
        """

        self.model_configuration = model_configuration

        self.cache = {}

    def get_default_boxes_matrix(self, image_shape):
        """
        Gets default boxes matrix for whole SSD model - made out of concatenations of
        default boxes matrices for different heads
        :param image_shape: tuple of ints, with first two elements being image's height and width
        :return: 2D numpy array
        """

        # If no cached matrix is present, compute one and store it in cache
        if image_shape not in self.cache.keys():

            default_boxes_matrices = []

            for prediction_head in self.model_configuration["prediction_heads_order"]:

                single_head_default_boxes_matrix = self.get_default_boxes_matrix_for_single_prediction_head(
                    self.model_configuration[prediction_head], image_shape)

                default_boxes_matrices.append(single_head_default_boxes_matrix)

            self.cache[image_shape] = np.concatenate(default_boxes_matrices)

        # Serve matrix from cache
        return self.cache[image_shape]

    @staticmethod
    def get_default_boxes_matrix_for_single_prediction_head(configuration, image_shape):
        """
        Gets default boxes matrix for a single prediction head
        :param configuration: dictionary of configuration options to be used to make default boxes matrix
        :param image_shape: tuple of ints, with first two elements being image's height and width
        :return: 2D numpy array
        """

        step = configuration["image_downscale_factor"]

        boxes_matrices = []

        for base_size in configuration["base_bounding_box_sizes"]:

            # Vertical boxes
            for aspect_ratio in configuration["aspect_ratios"]:

                width = aspect_ratio * base_size
                height = base_size

                boxes_matrix = DefaultBoxesFactory.get_single_configuration_boxes_matrix(
                    image_shape, step, width, height)

                boxes_matrices.append(boxes_matrix)

            # Horizontal boxes
            for aspect_ratio in configuration["aspect_ratios"]:

                width = base_size
                height = aspect_ratio * base_size

                boxes_matrix = DefaultBoxesFactory.get_single_configuration_boxes_matrix(
                    image_shape, step, width, height)

                boxes_matrices.append(boxes_matrix)

        return np.concatenate(boxes_matrices)

    @staticmethod
    def get_single_configuration_boxes_matrix(image_shape, step, width, height):
        """
        Gets default bounding boxes matrix for a single configuration
        :param image_shape: tuple (height, width)
        :param step: int, distance between neighbouring boxes
        :param width: float, box width
        :param height: float, box height
        :return: 2D numpy array
        """

        # First get a vector of centers in x and y directions
        y_centers = np.arange(step // 2, image_shape[0], step)
        x_centers = np.arange(step // 2, image_shape[1], step)

        y_steps = len(y_centers)
        x_steps = len(x_centers)

        # Now repeat x and y centers to create (x, y) paris for each case, and cast both to column vectors
        y_centers = np.repeat(y_centers, x_steps).reshape(-1, 1)
        x_centers = np.tile(x_centers, y_steps).reshape(-1, 1)

        half_width = width / 2
        half_height = height / 2

        return np.concatenate(
            [x_centers - half_width, y_centers - half_height, x_centers + half_width, y_centers + half_height],
            axis=1)


def get_matching_analysis_generator(ssd_model_configuration, ssd_input_generator):
    """
    Generator that accepts ssd_input_generator and yield a generator that outputs tuples of
    matched_annotations and unmatched_annotations, both of which are lists of Annotation instances.
    :param ssd_model_configuration: dictionary of options specifying ssd model's configuration
    :param ssd_input_generator: generator that outputs (image, annotations) tuples
    :return: generator that outputs (matched_annotations, unmatched_annotations) tuples
    """

    default_boxes_factory = DefaultBoxesFactory(ssd_model_configuration)

    while True:

        image, annotations = next(ssd_input_generator)
        default_boxes_matrix = default_boxes_factory.get_default_boxes_matrix(image.shape)

        matched_annotations = []
        unmatched_annotations = []

        for annotation in annotations:

            matched_default_boxes_indices = net.utilities.get_matched_boxes_indices(
                annotation.bounding_box, default_boxes_matrix)

            if len(matched_default_boxes_indices) > 0:

                matched_annotations.append(annotation)

            else:

                unmatched_annotations.append(annotation)

        yield matched_annotations, unmatched_annotations


class SSDTrainingLoopDataLoader:
    """
    Data loader class that outputs (image, indices of default boxes that matched annotations in image),
    or data suitable for training and evaluating SSD network
    """

    def __init__(self, voc_samples_data_loader, ssd_model_configuration):
        """
        Constructor
        :param voc_samples_data_loader: net.data.VOCSamplesDataLoader instance
        """

        self.voc_samples_data_loader = voc_samples_data_loader

        self.default_boxes_factory = DefaultBoxesFactory(ssd_model_configuration)

    def __len__(self):
        return len(self.voc_samples_data_loader)

    def __iter__(self):
        iterator = iter(self.voc_samples_data_loader)

        while True:

            image, annotations = next(iterator)
            default_boxes_matrix = self.default_boxes_factory.get_default_boxes_matrix(image.shape)

            matched_default_boxes_indices_set = set()

            for annotation in annotations:

                matched_default_boxes_indices = net.utilities.get_matched_boxes_indices(
                    annotation.bounding_box, default_boxes_matrix)

                matched_default_boxes_indices_set.update(matched_default_boxes_indices)

            # sorted() both sorts data and casts set into list, which can be converted to numpy array
            yield image, np.array(sorted(matched_default_boxes_indices_set)).astype(np.int32)

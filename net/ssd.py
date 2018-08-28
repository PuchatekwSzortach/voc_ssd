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

    def get_default_boxes_matrix(self, image_shape):
        """
        Gets default boxes matrix for whole SSD model - made out of concatenations of
        default boxes matrices for different heads
        :param image_shape: tuple of ints, with first two elements being image's height and width
        :return: 2D numpy array
        """

        default_boxes_matrices = []

        for prediction_head in self.model_configuration["predicion_heads_order"]:

            single_head_default_boxes_matrix = self.get_default_boxes_matrix_for_single_prediction_head(
                self.model_configuration[prediction_head], image_shape)

            default_boxes_matrices.append(single_head_default_boxes_matrix)

        return np.concatenate(default_boxes_matrices)

    @staticmethod
    def get_default_boxes_matrix_for_single_prediction_head(configuration, image_shape):
        """
        Gets default boxes matrix for a single prediction head
        :param configuration: dictionary of configuration options to be used to make default boxes matrix
        :param image_shape: tuple of ints, with first two elements being image's height and width
        :return: 2D numpy array
        """

        step = configuration["image_downscale_factor"]

        half_width = configuration["base_bounding_box_size"] / 2
        half_height = configuration["base_bounding_box_size"] / 2

        boxes = []

        for y in range(step // 2, image_shape[0], step):
            for x in range(step // 2, image_shape[1], step):

                box = [x - half_width, y - half_height, x + half_width, y + half_height]
                boxes.append(box)

        return np.vstack(boxes)


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

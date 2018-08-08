"""
Tests for net.utilities module
"""

import numpy as np

import net.utilities


def test_get_resized_sample_input_is_already_a_multiple_of_desired_factor():
    """
    Test for net.utilities.get_resized_sample(), checks input isn't changed when it already is a multiple
    of requested size factor
    """

    image = np.zeros((30, 60))
    bounding_boxes = [(10, 15, 40, 25)]
    size_factor = 10

    resized_image, resized_bounding_boxes = net.utilities.get_resized_sample(image, bounding_boxes, size_factor)

    assert image.shape == resized_image.shape
    assert bounding_boxes == resized_bounding_boxes


def test_get_resized_sample_input_is_not_a_multiple_of_desired_factor():
    """
    Test for net.utilities.get_resized_sample(), checks input is resized correctly as needed
    """

    image = np.zeros((30, 50))
    bounding_boxes = [(8, 10, 32, 17)]
    size_factor = 20

    expected_resized_image_shape = (20, 40)
    expected_resized_bounding_boxes = [(6, 7, 26, 11)]

    actual_resized_image, actual_resized_bounding_boxes = \
        net.utilities.get_resized_sample(image, bounding_boxes, size_factor)

    assert expected_resized_image_shape == actual_resized_image.shape
    assert expected_resized_bounding_boxes == actual_resized_bounding_boxes

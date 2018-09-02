"""
Tests for net.ssd module
"""

import numpy as np

import net.ssd


def test_get_single_configuration_boxes_matrix_square_boxes():
    """
    Test for creation of default boxes matrix for a single configuration.
    Aspect ratio of 1 is used, leading to square boxes.
    """

    image_shape = 4, 8
    step = 2
    base_size = 4
    aspect_ratio = 1

    expected = np.array([
        [-1, -1, 3, 3],
        [1, -1, 5, 3],
        [3, -1, 7, 3],
        [5, -1, 9, 3],
        [-1, 1, 3, 5],
        [1, 1, 5, 5],
        [3, 1, 7, 5],
        [5, 1, 9, 5],
    ])

    actual = net.ssd.DefaultBoxesFactory.get_single_configuration_boxes_matrix(
        image_shape, step, base_size, aspect_ratio)

    assert np.all(expected == actual)


def test_get_single_configuration_boxes_matrix_boxes_height_lower_than_width():
    """
    Test for creation of default boxes matrix for a single configuration.
    Aspect ratio of 2 is used, leading to boxes wider than taller.
    """

    image_shape = 12, 8
    step = 4
    base_size = 4
    aspect_ratio = 2

    # Boxes should have width of 8 and height of 4
    expected = np.array([
        [-2, 0, 6, 4],
        [2, 0, 10, 4],
        [-2, 4, 6, 8],
        [2, 4, 10, 8],
        [-2, 8, 6, 12],
        [2, 8, 10, 12],
    ])

    actual = net.ssd.DefaultBoxesFactory.get_single_configuration_boxes_matrix(
        image_shape, step, base_size, aspect_ratio)

    assert np.all(expected == actual)

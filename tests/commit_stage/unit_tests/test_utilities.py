"""
Tests for net.utilities module
"""

import numpy as np

import net.utilities


def test_get_target_size_round_down():
    """
    Test get_target_shape for a sample input that should be rounded down
    """

    image_size = 100, 200
    size_factor = 32

    expected = 96, 192
    actual = net.utilities.get_target_shape(image_size, size_factor)

    assert expected == actual


def test_get_target_size_round_up():
    """
    Test get_target_shape for a sample input that should be rounded up
    """

    image_size = 7, 3
    size_factor = 4

    expected = 8, 4
    actual = net.utilities.get_target_shape(image_size, size_factor)

    assert expected == actual


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


def test_is_annotation_size_unusual_object_has_too_small_width():
    """
    Test that checks correct response when object has small width
    """

    annotation = net.utilities.Annotation([10, 20, 11, 30], "")

    min_size = 4
    min_aspect_ratio = 0.2
    max_aspect_ratio = 5

    expected = True

    actual = net.utilities.is_annotation_size_unusual(
        annotation, min_size, min_aspect_ratio, max_aspect_ratio)

    assert expected == actual


def test_is_annotation_size_unusual_object_has_too_small_height():
    """
    Test that checks correct response when object has small width
    """

    annotation = net.utilities.Annotation([10, 20, 30, 22], "")

    min_size = 4
    min_aspect_ratio = 0.2
    max_aspect_ratio = 5

    expected = True

    actual = net.utilities.is_annotation_size_unusual(
        annotation, min_size, min_aspect_ratio, max_aspect_ratio)

    assert expected == actual


def test_is_annotation_size_unusual_object_has_too_small_aspect_ratio():
    """
    Test that checks correct response when object has small width
    """

    annotation = net.utilities.Annotation([10, 20, 20, 1000], "")

    min_size = 2
    min_aspect_ratio = 0.2
    max_aspect_ratio = 5

    expected = True

    actual = net.utilities.is_annotation_size_unusual(
        annotation, min_size, min_aspect_ratio, max_aspect_ratio)

    assert expected == actual


def test_is_annotation_size_unusual_object_has_too_large_aspect_ratio():
    """
    Test that checks correct response when object has small width
    """

    annotation = net.utilities.Annotation([10, 20, 210, 30], "")
    min_size = 2
    min_aspect_ratio = 0.01
    max_aspect_ratio = 5

    expected = True

    actual = net.utilities.is_annotation_size_unusual(
        annotation, min_size, min_aspect_ratio, max_aspect_ratio)

    assert expected == actual


def test_is_annotation_size_unusual_object_has_normal_size():
    """
    Test that checks correct response when object has small width
    """

    annotation = net.utilities.Annotation([10, 20, 30, 50], "")
    min_size = 10
    min_aspect_ratio = 0.2
    max_aspect_ratio = 5

    expected = False

    actual = net.utilities.is_annotation_size_unusual(
        annotation, min_size, min_aspect_ratio, max_aspect_ratio)

    assert expected == actual

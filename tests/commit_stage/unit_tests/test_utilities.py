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


def test_annotation_resize_no_resize_needed():
    """
    Test Annotation.resize - no resize actually needed
    """

    annotation = net.utilities.Annotation([10, 20, 20, 40])

    image_size = (100, 200)
    size_factor = 10

    expected = net.utilities.Annotation([10, 20, 20, 40])
    actual = annotation.resize(image_size, size_factor)

    assert expected == actual


def test_annotation_resize_resize_needed():
    """
    Test Annotation.resize - both shrinking and expanding object needed
    """

    annotation = net.utilities.Annotation([10, 20, 20, 40])

    image_size = (80, 220)
    size_factor = 50

    expected = net.utilities.Annotation([9, 25, 18, 50])
    actual = annotation.resize(image_size, size_factor)

    assert expected == actual


def test_round_to_factor_factor_is_an_integer_rounding_up_is_needed():
    """
    Test round_to_factor when rounding up to an integer factor
    """

    value = 18
    factor = 5

    expected = 20
    actual = net.utilities.round_to_factor(value, factor)

    assert expected == actual


def test_round_to_factor_factor_is_an_integer_rounding_down_is_needed():
    """
    Test round_to_factor when rounding down to an integer factor
    """

    value = 17
    factor = 5

    expected = 15
    actual = net.utilities.round_to_factor(value, factor)

    assert expected == actual


def test_round_to_factor_factor_is_a_float_rounding_up_is_needed():
    """
    Test round_to_factor when rounding up to a float factor
    """

    value = 10.91
    factor = 0.2

    expected = 11
    actual = net.utilities.round_to_factor(value, factor)

    assert expected == actual


def test_round_to_factor_factor_is_a_float_rounding_down_is_needed():
    """
    Test round_to_factor when rounding down to a float factor
    """

    value = 10.87
    factor = 0.2

    expected = 10.8
    actual = net.utilities.round_to_factor(value, factor)

    assert expected == actual


def test_get_annotations_from_default_boxes():
    """
    Test converting default boxes matrix to list of Annotation instances
    """

    default_boxes_matrix = np.array([
        [10, 20, 30, 40, 0],
        [40, 20, 80, 120, 2],
        [1, 200, 300, 240, 1],
    ])

    expected = [
        net.utilities.Annotation([10, 20, 30, 40]),
        net.utilities.Annotation([40, 20, 80, 120]),
        net.utilities.Annotation([1, 200, 300, 240])
    ]

    actual = net.utilities.get_annotations_from_default_boxes(default_boxes_matrix)

    assert expected == actual


def test_get_matched_boxes_indices_decision_can_be_made_looking_at_borders():
    """
    Test code matching matrix of bounding boxes with a template box.
    Only simple input is used - boxes either hardly overlap, or overlap with high IOU.
    """

    template_box = (100, 100, 200, 200)

    boxes_matrix = np.array([
        [50, 50, 140, 140],
        [101, 101, 200, 200],
        [170, 170, 200, 200],
        [99, 99, 201, 201],

    ])

    expected = np.array([1, 3])
    actual = net.utilities.get_matched_boxes_indices(template_box, boxes_matrix)

    assert np.all(expected == actual)


def test_get_matched_boxes_indices_decision_can_be_made_looking_boxes_sizes_default_boxes_inside_template_box():
    """
    Test code matching matrix of bounding boxes with a template box.
    All default boxes are inside template box, but some have too small area to have high IOU
    """

    template_box = (100, 100, 200, 200)

    boxes_matrix = np.array([
        [101, 101, 199, 199],
        [140, 140, 160, 160],
        [102, 102, 198, 198],
        [145, 145, 155, 155],

    ])

    expected = np.array([0, 2])
    actual = net.utilities.get_matched_boxes_indices(template_box, boxes_matrix)

    assert np.all(expected == actual)


def test_get_matched_boxes_boxes_indices_iou_has_to_be_calculated_to_make_decision():
    """
    Test code matching matrix of bounding boxes with a template box.
    All boxes overlap at least partially and iou has to be calculated to decide if boxes match with template
    """

    template_box = (100, 100, 200, 200)

    boxes_matrix = np.array([
        [110, 100, 210, 210],
        [80, 80, 190, 190],
        [70, 60, 180, 170],
        [70, 60, 200, 200],

    ])

    expected = np.array([0, 1, 3])
    actual = net.utilities.get_matched_boxes_indices(template_box, boxes_matrix)

    assert np.all(expected == actual)


def test_get_matched_boxes_boxes_indices_some_boxes_can_be_discarded_early_some_need_iou_computations():
    """
    Test code matching matrix of bounding boxes with a template box.
    Some boxes can be discarded with simple checks, but some required explicit IOU computations.
    """

    template_box = (100, 100, 200, 200)

    boxes_matrix = np.array([
        [110, 110, 210, 210],
        [50, 50, 70, 90],
        [170, 80, 220, 200],
        [95, 105, 200, 205]
    ])

    expected = np.array([0, 3])
    actual = net.utilities.get_matched_boxes_indices(template_box, boxes_matrix)

    assert np.all(expected == actual)


def test_get_vectorized_intersection_over_union_all_boxes_are_outside_of_template_box():
    """
    Test vectorized iou computations.
    Boxes don't overlap with template box.
    """

    template_box = (100, 100, 200, 200)

    boxes_matrix = np.array([
        [10, 10, 20, 20],  # Above and to the left of template box
        [50, 100, 80, 200],  # Same y-coordinates, but to the left to template box
        [250, 100, 270, 200],  # Same y-coordinates, but to the right to template box
        [100, 40, 200, 90],  # Same x-coordinates, but above the template box
        [100, 230, 200, 300],  # Same x-coordinates, but below the template box
        [300, 300, 400, 400]  # Below and to the right of the template box
    ])

    expected = np.array([0, 0, 0, 0, 0, 0])
    actual = net.utilities.get_vectorized_intersection_over_union(template_box, boxes_matrix)

    assert np.all(expected == actual)


def test_get_vectorized_intersection_over_various_overlapping_boxes():
    """
    Test vectorized iou computations.
    Boxes overlap with template box to various degrees.
    """

    template_box = (100, 100, 200, 200)

    boxes_matrix = np.array([
        [100, 100, 200, 200],
        [50, 150, 180, 220],
        [80, 70, 170, 200],
        [160, 180, 300, 300],

    ])

    expected = np.array([1, 40/151, 70/147, 2/65])
    actual = net.utilities.get_vectorized_intersection_over_union(template_box, boxes_matrix)

    assert np.all(expected == actual)

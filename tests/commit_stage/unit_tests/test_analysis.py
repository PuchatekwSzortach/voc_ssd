"""
Tests for analysis module
"""

import numpy as np

import net.analysis
import net.utilities


def test_get_heatmap_all_data_in_same_bin():
    """
    Test get_heatmap with all data in the same bin
    """

    data = np.array([
        [10, 10],
        [10, 10],
        [10, 10]
    ])

    expected = np.array([
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 3, 0],
        [0, 0, 0, 0]
    ])

    actual = net.analysis.get_heatmap(data=data, bin_size=5, max_size=20)

    assert actual.shape == (4, 4)
    assert np.all(expected == actual)


def test_get_heatmap_data_in_different_bins():
    """
    Test get_heatmap with data falling into different bins
    """

    data = np.array([
        [10, 10],
        [3, 10],
        [19, 6]
    ])

    expected = np.array([
        [0, 0, 1, 0],
        [0, 0, 0, 0],
        [0, 0, 1, 0],
        [0, 1, 0, 0]
    ])

    actual = net.analysis.get_heatmap(data=data, bin_size=5, max_size=20)

    assert actual.shape == (4, 4)
    assert np.all(expected == actual)


def test_get_heatmap_data_exceeding_max_size():
    """
    Test get_heatmap with some data exceeding max size
    """

    data = np.array([
        [50, 10],
        [3, 10],
        [9, 60]
    ])

    expected = np.array([
        [0, 0, 1, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0]
    ])

    actual = net.analysis.get_heatmap(data=data, bin_size=5, max_size=20)

    assert actual.shape == (4, 4)
    assert np.all(expected == actual)


def test_get_predictions_matches():
    """
    Test get_predictions_matches on simple inputs
    """

    ground_truth_annotations = [
        net.utilities.Annotation(bounding_box=[10, 10, 100, 100], label="car"),
        net.utilities.Annotation(bounding_box=[20, 50, 80, 120], label="dog"),
        net.utilities.Annotation(bounding_box=[30, 50, 200, 300], label="airplane")
    ]

    predictions = [
        net.utilities.Prediction(bounding_box=[20, 50, 80, 120], confidence=0.8, label="dog"),
        net.utilities.Prediction(bounding_box=[20, 50, 80, 120], confidence=0.9, label="dog"),
        net.utilities.Prediction(bounding_box=[10, 10, 100, 100], confidence=0.7, label="car")
    ]

    expected = [
        {
            "prediction": predictions[1],
            "is_correct": True,
        },
        {
            "prediction": predictions[0],
            "is_correct": False,
        },
        {
            "prediction": predictions[2],
            "is_correct": True,
        },
    ]

    actual = net.analysis.get_predictions_matches(ground_truth_annotations, predictions)

    assert expected == actual

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


def test_get_predictions_matches_on_simple_inputs():
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


def test_get_predictions_matches_with_multiple_predictions_for_the_same_ground_truth_annotation():
    """
    Test get_predictions_matches with multiple predictions for the same ground truth annotation
    """

    ground_truth_annotations = [
        net.utilities.Annotation(bounding_box=[10, 10, 100, 100], label="car"),
    ]

    predictions = [
        net.utilities.Prediction(bounding_box=[10, 10, 100, 100], confidence=0.8, label="car"),
        net.utilities.Prediction(bounding_box=[10, 10, 100, 100], confidence=0.9, label="car"),
        net.utilities.Prediction(bounding_box=[10, 10, 100, 100], confidence=0.7, label="car"),
        net.utilities.Prediction(bounding_box=[10, 10, 100, 100], confidence=0.6, label="car")
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
            "is_correct": False,
        },
        {
            "prediction": predictions[3],
            "is_correct": False,
        }
    ]

    actual = net.analysis.get_predictions_matches(ground_truth_annotations, predictions)

    assert expected == actual


class TestMeanAveragePrecisionComputer:
    """
    Tests for MeanAveragePrecisionComputer class
    """

    def test_get_recall_values_and_precision_values_data(self):
        """
        Test for _get_recall_values_and_precision_values_data function
        """

        predictions_matches_data = [
            {
                "prediction": net.utilities.Prediction(bounding_box=[20, 50, 80, 120], confidence=0.9, label="dog"),
                "is_correct": True,
            },
            {
                "prediction": net.utilities.Prediction(bounding_box=[20, 50, 80, 120], confidence=0.8, label="dog"),
                "is_correct": False,
            },
            {
                "prediction": net.utilities.Prediction(bounding_box=[10, 10, 100, 100], confidence=0.7, label="car"),
                "is_correct": True,
            },
        ]

        expected_recall_values = [0.25, 0.25, 0.5]
        expected_precision_values = [1, 0.5, 2/3]

        actual_recall_values, actual_precision_values = \
            net.analysis.MeanAveragePrecisionComputer.get_recall_values_and_precision_values_data(
                predictions_matches_data=predictions_matches_data,
                ground_truth_annotations_count=4)

        assert np.all(expected_recall_values == actual_recall_values)
        assert np.all(expected_precision_values == actual_precision_values)

    def test_get_smoothed_out_precision_values(self):
        """
        Test get_smoothed_out_precision_values function
        """

        precision_values = [1, 1, 0.5, 0.7, 0.3, 0.5]

        expected = [1, 1, 0.7, 0.7, 0.5, 0.5]

        actual = net.analysis.MeanAveragePrecisionComputer.get_smoothed_out_precision_values(
            precision_values=precision_values)

        assert np.all(expected == actual)

    def test_get_interpolated_precision_values_complete_recall_values_available(self):
        """
        Test get_interpolated_precision_values with input such that there are precision values available for
        a full range of recall values
        """

        recall_values = [0, 0.05, 0.25, 0.75, 1]
        precision_values = [0.5, 0.7, 0.8, 0.4, 0.2]

        expected = [
            0.5,  # At recall 0
            0.725,  # At recall 0.1
            0.775,  # At recall 0.2
            0.76,  # At recall 0.3
            0.68,  # At recall 0.4
            0.6,  # At recall 0.5
            0.52,  # At recall 0.6
            0.44,  # At recall 0.7
            0.36,  # At recall 0.8
            0.28,  # At recall 0.9
            0.2  # At recall 1
        ]

        actual = net.analysis.MeanAveragePrecisionComputer.get_interpolated_precision_values(
            recall_values=recall_values,
            precision_values=precision_values)

        assert np.allclose(expected, actual)

    def test_get_interpolated_precision_values_no_high_recall_values_available(self):
        """
        Test get_interpolated_precision_values with input such that there recall values don't go all the way
        up to 1.
        """

        recall_values = [0, 0.05, 0.25, 0.75]
        precision_values = [0.5, 0.7, 0.8, 0.4]

        expected = [
            0.5,  # At recall 0
            0.725,  # At recall 0.1
            0.775,  # At recall 0.2
            0.76,  # At recall 0.3
            0.68,  # At recall 0.4
            0.6,  # At recall 0.5
            0.52,  # At recall 0.6
            0.44,  # At recall 0.7
            0,  # At recall 0.8
            0,  # At recall 0.9
            0  # At recall 1
        ]

        actual = net.analysis.MeanAveragePrecisionComputer.get_interpolated_precision_values(
            recall_values=recall_values,
            precision_values=precision_values)

        assert np.allclose(expected, actual)


def test_get_unique_prediction_matches():
    """
    Test for get_unique_prediction_matches function
    """

    ground_truths = [
        net.utilities.Annotation(bounding_box=[10, 10, 100, 100], label="car"),
        net.utilities.Annotation(bounding_box=[20, 50, 80, 120], label="dog"),
        net.utilities.Annotation(bounding_box=[30, 50, 200, 300], label="airplane")
    ]

    predictions = [
        net.utilities.Prediction(bounding_box=[20, 50, 80, 120], confidence=0.9, label="dog"),
        net.utilities.Prediction(bounding_box=[20, 50, 80, 120], confidence=0.8, label="dog"),
        net.utilities.Prediction(bounding_box=[10, 10, 100, 100], confidence=0.7, label="car")
    ]

    expected = [
        net.utilities.Prediction(bounding_box=[20, 50, 80, 120], confidence=0.9, label="dog"),
        net.utilities.Prediction(bounding_box=[10, 10, 100, 100], confidence=0.7, label="car")
    ]

    actual = net.analysis.get_unique_prediction_matches(ground_truths, predictions)

    assert expected == actual

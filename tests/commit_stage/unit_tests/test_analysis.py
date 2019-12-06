"""
Tests for analysis module
"""

import numpy as np

import net.analysis


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

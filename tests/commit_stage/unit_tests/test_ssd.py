"""
Tests for net.ssd module
"""

import numpy as np
import tensorflow as tf

import net.ssd


def test_get_single_shot_detector_loss_op_with_no_positives_matches():
    """
    Test ssd loss op when no positive match was present - means loss should be 0
    """

    default_boxes_categories_ids_vector = np.array([0, 0])

    predictions_logits_matrix = np.array([
        [0.8, 0.5],
        [0.4, 0.2]
    ])

    loss_op = net.ssd.get_single_shot_detector_loss_op(
        default_boxes_categories_ids_vector_op=tf.constant(default_boxes_categories_ids_vector, dtype=tf.int32),
        predictions_logits_matrix_op=tf.constant(predictions_logits_matrix, dtype=tf.float32),
        hard_negatives_mining_ratio=3)

    with tf.Session() as session:

        loss = session.run(loss_op)

        assert loss == 0


def test_get_single_shot_detector_loss_op_with_no_hard_negatives_mining():
    """
    Test ssd loss op when no negative samples are used
    """

    default_boxes_categories_ids_vector = np.array([1, 0])

    predictions_logits_matrix = np.array([
        [0.5, 0.1],
        [0.4, 0.2]
    ])

    loss_op = net.ssd.get_single_shot_detector_loss_op(
        default_boxes_categories_ids_vector_op=tf.constant(default_boxes_categories_ids_vector, dtype=tf.int32),
        predictions_logits_matrix_op=tf.constant(predictions_logits_matrix, dtype=tf.float32),
        hard_negatives_mining_ratio=0)

    with tf.Session() as session:

        expected = -np.log(0.4013)
        actual = session.run(loss_op)

        assert np.isclose(expected, actual, atol=0.001)


def test_get_single_shot_detector_loss_op_with_all_samples_used():
    """
    Test ssd loss op when all positive and negative samples are used
    """

    default_boxes_categories_ids_vector = np.array([1, 0])

    predictions_logits_matrix = np.array([
        [0.5, 0.1],
        [0.4, 0.2]
    ])

    loss_op = net.ssd.get_single_shot_detector_loss_op(
        default_boxes_categories_ids_vector_op=tf.constant(default_boxes_categories_ids_vector, dtype=tf.int32),
        predictions_logits_matrix_op=tf.constant(predictions_logits_matrix, dtype=tf.float32),
        hard_negatives_mining_ratio=1)

    with tf.Session() as session:

        expected = -(np.log(0.4013) + np.log(0.5498)) / 2.0
        actual = session.run(loss_op)

        assert np.isclose(expected, actual, atol=0.001)


def test_get_single_shot_detector_loss_op_with_not_all_negative_samples_used():
    """
    Test ssd loss op when not all negative samples are used
    """

    default_boxes_categories_ids_vector = np.array([1, 0, 0])

    predictions_logits_matrix = np.array([
        [0.5, 0.1],
        [0.4, 0.2],
        [0.8, 0.2],
    ])

    loss_op = net.ssd.get_single_shot_detector_loss_op(
        default_boxes_categories_ids_vector_op=tf.constant(default_boxes_categories_ids_vector, dtype=tf.int32),
        predictions_logits_matrix_op=tf.constant(predictions_logits_matrix, dtype=tf.float32),
        hard_negatives_mining_ratio=1)

    with tf.Session() as session:

        expected = -(np.log(0.4013) + np.log(0.5498)) / 2.0
        actual = session.run(loss_op)

        assert np.isclose(expected, actual, atol=0.001)


def test_get_single_shot_detector_loss_op_with_complex_data():
    """
    Test ssd loss op with complex data - multiple positives, multiple negatives, not all data is used
    """

    default_boxes_categories_ids_vector = np.array([2, 0, 0, 0, 0, 0, 0, 1])

    predictions_logits_matrix = np.array([
        [0.5, 0.1, 0.3],  # softmax on correct label is 0.3289
        [0.4, 0.2, 0.1],  # softmax on correct label is 0.3907
        [0.8, 0.2, 0.2],  # softmax on correct label is 0.4767
        [0.1, 0.8, 0.2],  # softmax on correct label is 0.2428
        [0.8, 0.1, 0.1],  # softmax on correct label is 0.5017
        [0.3, 0.1, 0.1],  # softmax on correct label is 0.3792
        [0.5, 0.3, 0.9],  # softmax on correct label is 0.3021
        [0.5, 0.1, 0.9],  # softmax on correct label is 0.2119
    ])

    loss_op = net.ssd.get_single_shot_detector_loss_op(
        default_boxes_categories_ids_vector_op=tf.constant(default_boxes_categories_ids_vector, dtype=tf.int32),
        predictions_logits_matrix_op=tf.constant(predictions_logits_matrix, dtype=tf.float32),
        hard_negatives_mining_ratio=2)

    with tf.Session() as session:

        # We expect losses for two positive boxes and losses for 4 negative boxes with worst prediction to be used
        positive_labels = [0.3289, 0.2119]
        hard_negative_mining_labels = [0.2428, 0.3021, 0.3792, 0.3907]

        expected = np.mean(-np.log(positive_labels + hard_negative_mining_labels))
        actual = session.run(loss_op)

        assert np.isclose(expected, actual, atol=0.001)

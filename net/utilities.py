"""
Module with various utilities
"""

import os
import logging

import cv2


def get_logger(path):
    """
    Returns a logger that writes to an html page
    :param path: path to log.html page
    :return: logger instance
    """

    os.makedirs(os.path.dirname(path), exist_ok=True)

    logger = logging.getLogger("voc_fcn")
    file_handler = logging.FileHandler(path, mode="w")

    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)

    return logger


def get_image_with_bounding_boxes(image, bounding_boxes):
    """
    Get a copy of input image with bounding boxes drawn on it
    :param image: numpy array
    :param bounding_boxes: list of bounding boxes in [x_min, y_min, x_max, y_max] format
    :return: numpy array
    """

    annotated_image = image.copy()

    for box in bounding_boxes:

        cv2.rectangle(annotated_image, (box[0], box[1]), (box[2], box[3]), color=(0, 255, 0), thickness=2)

    return annotated_image

"""
Module with various utilities
"""

import os
import logging

import numpy as np
import PIL.ImageDraw
import PIL.ImageFont
import PIL.Image
import cv2


class Annotation:
    """
    A simple class for bundling bounding box and category of an object
    """

    def __init__(self, bounding_box, label):

        self.bounding_box = bounding_box
        self.label = label


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


def get_ids_to_colors_map(categories_count):
    """
    Given a categories count, get categories ids to colors dictionary. Colors are computed as a function of
    category index.
    All colors are returned in BGR order.
    Code adapted from https://gist.github.com/wllhf/a4533e0adebe57e3ed06d4b50c8419ae
    :param categories_count: number of categories
    :return: map {int: tuple}
    """

    colors_count = 256

    def bitget(byte_value, idx):
        """
        Check if bit at given byte index is set
        :param byte_value: byte
        :param idx: index
        :return: bool
        """
        return (byte_value & (1 << idx)) != 0

    colors_matrix = np.zeros(shape=(colors_count, 3), dtype=np.int)

    for color_index in range(colors_count):

        red = green = blue = 0
        color = color_index

        for j in range(8):

            red = red | (bitget(color, 0) << 7 - j)
            green = green | (bitget(color, 1) << 7 - j)
            blue = blue | (bitget(color, 2) << 7 - j)
            color = color >> 3

        # Writing colors in BGR order
        colors_matrix[color_index] = blue, green, red

    # Return category index to colors map, make sure to convert each color from numpy array to plain list
    return {category_index: colors_matrix[category_index].tolist() for category_index in range(categories_count)}


def get_categories_to_colors_map(categories):
    """
    Given a list of categories, get categories to colors dictionary. Colors are computed as a function of
    category index.
    All colors are returned in BGR order.
    Code adapted from https://gist.github.com/wllhf/a4533e0adebe57e3ed06d4b50c8419ae
    :param categories: list of strings
    :return: map {string: tuple}
    """

    ids_to_colors_map = get_ids_to_colors_map(len(categories))
    return {categories[index]: ids_to_colors_map[index] for index in range(len(categories))}


def get_image_with_bounding_boxes(image, bounding_boxes):
    """
    Get a copy of input image with bounding boxes drawn on it
    :param image: numpy array
    :param bounding_boxes: list of bounding boxes in [x_min, y_min, x_max, y_max] format
    :return: numpy array, an annotated image
    """

    annotated_image = image.copy()

    for box in bounding_boxes:

        cv2.rectangle(annotated_image, (box[0], box[1]), (box[2], box[3]), color=(0, 255, 0), thickness=2)

    return annotated_image


def draw_annotation_label(pil_image, annotation, color, font):
    """
    Draw annotation label on an image
    :param pil_image: PIL.Image object
    :param annotation: net.utilities.Annotation instance
    :param color: color to use for label background
    :param font: PIL.ImageFont object
    """

    draw_manager = PIL.ImageDraw.Draw(pil_image)

    text_size = draw_manager.textsize(annotation.label, font)

    box_left = int(max(0, np.floor(annotation.bounding_box[0] + 0.5)))
    box_top = int(max(0, np.floor(annotation.bounding_box[1] + 0.5)))

    text_origin = [box_left, box_top - text_size[1]] \
        if box_top - text_size[1] >= 0 else [box_left, box_top + 1]

    text_end = text_origin[0] + text_size[0], text_origin[1] + text_size[1]
    text_box = text_origin[0], text_origin[1], text_end[0], text_end[1]

    draw_manager.rectangle(text_box, fill=tuple(color))
    draw_manager.text(text_origin, annotation.label, fill=(255, 255, 255), font=font)


def get_annotated_image(image, annotations, colors, font_path):
    """
    Get a copy of input image with bounding boxes drawn on it. Colors of bounding boxes are selected per
    unique per category and each bounding box includes a label stating its category.
    :param image: numpy array
    :param annotations: list of net.utilities.Annotation object
    :param colors: list of colors to be used for each annotation
    :param font_path: path to font file
    :return: numpy array, an annotated image
    """

    annotated_image = image.copy()

    for annotation, color in zip(annotations, colors):

        box = annotation.bounding_box

        cv2.rectangle(annotated_image, (box[0], box[1]), (box[2], box[3]), color=color, thickness=3)

    pil_image = PIL.Image.fromarray(annotated_image)

    font = PIL.ImageFont.truetype(font_path, size=20)

    for annotation, color in zip(annotations, colors):

        draw_annotation_label(pil_image, annotation, color, font)

    return np.array(pil_image)


def get_target_shape(shape, size_factor):
    """
    Given an shape tuple and size_factor, return a new shape tuple such that each of its dimensions
    is a multiple of size_factor, rounding down
    :param shape: tuple of integers
    :param size_factor: integer
    :return: tuple of integers
    """

    target_shape = []

    for dimension in shape:

        target_dimension = size_factor * (dimension // size_factor)
        target_shape.append(target_dimension)

    return tuple(target_shape)


def get_resized_sample(image, bounding_boxes, size_factor):
    """
    Resize image and its annotations so that image is a multiple of factor
    :param image: numpy array
    :param bounding_boxes: list bounding boxes in corner format
    :param size_factor: int, value a multiple of which we want image to be
    :return: tuple (resized_image, resized_annotations)
    """

    target_y_size, target_x_size = get_target_shape(image.shape[:2], size_factor)

    resized_image = cv2.resize(image, (target_x_size, target_y_size))

    x_resize_fraction = target_x_size / image.shape[1]
    y_resize_fraction = target_y_size / image.shape[0]

    resized_bounding_boxes = []

    for bounding_box in bounding_boxes:

        x_min, y_min, x_max, y_max = bounding_box

        resized_bounding_box = \
            round(x_min * x_resize_fraction), round(y_min * y_resize_fraction), \
            round(x_max * x_resize_fraction), round(y_max * y_resize_fraction)

        resized_bounding_boxes.append(resized_bounding_box)

    return resized_image, resized_bounding_boxes

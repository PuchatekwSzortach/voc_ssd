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


def draw_bounding_box_label(pil_image, bounding_box, label, color, font):
    """
    Draw bounding box label on an image
    :param pil_image: PIL.Image object
    :param bounding_box: bounding box in format [x_min, y_min, x_max, y_max]
    :param label: text to draw
    :param color: color to use for label background
    :param font: PIL.ImageFont object
    """

    draw_manager = PIL.ImageDraw.Draw(pil_image)

    text_size = draw_manager.textsize(label, font)

    box_left = int(max(0, np.floor(bounding_box[0] + 0.5)))
    box_top = int(max(0, np.floor(bounding_box[1] + 0.5)))

    text_origin = [box_left, box_top - text_size[1]] \
        if box_top - text_size[1] >= 0 else [box_left, box_top + 1]

    text_end = text_origin[0] + text_size[0], text_origin[1] + text_size[1]
    text_box = text_origin[0], text_origin[1], text_end[0], text_end[1]

    draw_manager.rectangle(text_box, fill=tuple(color))
    draw_manager.text(text_origin, label, fill=(255, 255, 255), font=font)


def get_annotated_image(image, bounding_boxes, categories, categories_to_colors_map, font_path):
    """
    Get a copy of input image with bounding boxes drawn on it. Colors of bounding boxes are selected per
    unique per category and each bounding box includes a label stating its category.
    :param image: numpy array
    :param bounding_boxes: list of bounding boxes in [x_min, y_min, x_max, y_max] format
    :param categories: list of categories - one for each bounding box
    :param categories_to_colors_map: categories to colors map
    :param font_path: path to font file
    :return: numpy array, an annotated image
    """

    annotated_image = image.copy()

    for box, category in zip(bounding_boxes, categories):

        cv2.rectangle(
            annotated_image, (box[0], box[1]), (box[2], box[3]),
            color=categories_to_colors_map[category], thickness=3)

    pil_image = PIL.Image.fromarray(annotated_image)

    font = PIL.ImageFont.truetype(font_path, size=20)

    for bounding_box, category in zip(bounding_boxes, categories):

        draw_bounding_box_label(pil_image, bounding_box, category, categories_to_colors_map[category], font)

    return np.array(pil_image)

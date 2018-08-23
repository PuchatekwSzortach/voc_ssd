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

    def __init__(self, bounding_box, label=None):

        self.bounding_box = tuple(bounding_box)
        self.label = label

    @property
    def width(self):
        """
        Width of annotation's bounding box
        """
        return self.bounding_box[2] - self.bounding_box[0]

    @property
    def height(self):
        """
        Height of annotation's bounding box
        """
        return self.bounding_box[3] - self.bounding_box[1]

    @property
    def aspect_ratio(self):
        """
        Aspect ratio of annotation's bounding box
        """
        return self.width / self.height

    @property
    def size(self):
        """
        height, width tuple
        """
        return self.height, self.width

    def resize(self, image_size, size_factor):
        """
        Returns a new Annotation instance that is resized as if hypothetical image containing it was resized to be
        a multiple of size_factor
        :param image_size: tuple of ints, (height, width)
        :param size_factor: int
        :return: Annotation instance
        """

        target_shape = get_target_shape(image_size, size_factor)

        y_resize_fraction = target_shape[0] / image_size[0]
        x_resize_fraction = target_shape[1] / image_size[1]

        x_min, y_min, x_max, y_max = self.bounding_box

        resized_bounding_box = \
            round(x_min * x_resize_fraction), round(y_min * y_resize_fraction), \
            round(x_max * x_resize_fraction), round(y_max * y_resize_fraction)

        return Annotation(resized_bounding_box, self.label)

    def __eq__(self, other):

        if not isinstance(other, self.__class__):
            return False

        return self.bounding_box == other.bounding_box and self.label == other.label

    def __repr__(self):

        return "Annotation: {}, {}".format(self.label, self.bounding_box)


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
    is a multiple of size_factor
    :param shape: tuple of integers
    :param size_factor: integer
    :return: tuple of integers
    """

    target_shape = []

    for dimension in shape:

        rounded_down_dimension = size_factor * (dimension // size_factor)
        rounded_up_dimension = rounded_down_dimension + size_factor

        rounded_down_difference = abs(dimension - rounded_down_dimension)
        rounded_up_difference = abs(dimension - rounded_up_dimension)

        target_dimension = \
            rounded_down_dimension if rounded_down_difference <= rounded_up_difference else rounded_up_dimension

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


def is_annotation_size_unusual(annotation, minimum_size, minimum_aspect_ratio, maximum_aspect_ratio):
    """
    Checks if object described by annotation has unusual size - is too small or has unusual aspect ratio
    :param annotation: net.utilities.Annotation instance
    :param minimum_size: int, minimum size object must have to be considered normal
    :param minimum_aspect_ratio: float, minimum aspect ratio object must have to be considered normal.
    Both width to height and height to width ratios are tested against this criterion
    :param maximum_aspect_ratio: float, maximum aspect ratio object must have to be considered normal.
    Both width to height and height to width ratios are tested against this criterion
    :return: bool, True if object size is unusual, False otherwise
    """

    if annotation.width < minimum_size or annotation.height < minimum_size:
        return True

    if annotation.aspect_ratio < minimum_aspect_ratio or 1 / annotation.aspect_ratio < minimum_aspect_ratio:
        return True

    if annotation.aspect_ratio > maximum_aspect_ratio or 1 / annotation.aspect_ratio > maximum_aspect_ratio:
        return True

    return False

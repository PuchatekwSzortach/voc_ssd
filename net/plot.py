""""
Module with plotting functionality
"""

import numpy as np
import PIL.ImageDraw
import PIL.Image
import PIL.ImageFont
import cv2

import net.utilities


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


def get_annotated_image(image, annotations, colors, draw_labels=True, font_path=None):
    """
    Get a copy of input image with bounding boxes drawn on it. Colors of bounding boxes are selected per
    unique per category and each bounding box includes a label stating its category.
    :param image: numpy array
    :param annotations: list of net.utilities.Annotation object
    :param colors: list of colors to be used for each annotation
    :param draw_labels: bool, specifies whether labels should be drawn
    :param font_path: path to font file
    :return: numpy array, an annotated image
    """

    annotated_image = image.copy()

    for annotation, color in zip(annotations, colors):

        box = annotation.bounding_box
        cv2.rectangle(annotated_image, (box[0], box[1]), (box[2], box[3]), color=color, thickness=3)

    # If we don't need to draw labels, finish processing here
    if draw_labels is False:
        return annotated_image

    # Else draw labels as well
    pil_image = PIL.Image.fromarray(annotated_image)
    font = PIL.ImageFont.truetype(font_path, size=20)

    for annotation, color in zip(annotations, colors):

        draw_annotation_label(pil_image, annotation, color, font)

    return np.array(pil_image)


def get_image_with_boxes(image, boxes, color):
    """
    Creates a new image with boxes drawn on original image in color specified
    :param image: numpy array
    :param boxes: list of boxes in corner format
    :param color: 3-element tuple
    :return: numpy array
    """

    boxes_annotations = net.utilities.get_annotations_from_default_boxes(boxes)
    colors = [color for _ in boxes]

    return net.plot.get_annotated_image(image, boxes_annotations, colors=colors, draw_labels=False)

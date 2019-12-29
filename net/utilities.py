"""
Module with various utilities
"""

import os
import logging
import collections

import numpy as np
import cv2


class Annotation:
    """
    A simple class for bundling bounding box and category of an object
    """

    def __init__(self, bounding_box, label=None, category_id=None):

        self.bounding_box = tuple(bounding_box)
        self.label = label
        self.category_id = category_id

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


class DefaultBoxDefinition:
    """
    Class holding definition of a default bounding box - its width, height and step to next box.
    """

    def __init__(self, width, height, step):
        """
        Constructor
        :param width: float, width of the box
        :param height: float, height of the box
        :param step: float, distance between two neighbouring boxes
        """

        self.width = width
        self.height = height
        self.step = step

    def __repr__(self):

        return "Default box definition: {}x{}, step of {}".format(self.width, self.height, self.step)

    def get_overlaps(self, other):
        """
        Computes intersection over union with other box definition placed at three different positions -
        centered same as self, moved by other box definitions's steps away from self's center horizontally,
        moved by other box definitions's steps away from self's center vertically
        :param other: DefaultBoxDefinition instance
        :return: dictionary with keys: center_iou, horizontal_shift_iou, vertical_shift_iou
        """

        template_box = [-self.width/2, -self.height/2, self.width/2, self.height/2]

        other_boxes = np.array([
            [-other.width/2, -other.height/2, other.width/2, other.height/2],
            [(-other.width/2) + other.step, -other.height/2, (other.width/2) + other.step, other.height/2],
            [-other.width/2, (-other.height/2) + other.step, other.width/2, (other.height/2) + other.step]
        ])

        ious = get_vectorized_intersection_over_union(template_box, other_boxes)

        return {
            "center_iou": ious[0],
            "horizontal_shift_iou": ious[1],
            "vertical_shift_iou": ious[2],
        }


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


def round_to_factor(value, factor):
    """
    Round value to nearest multiple of factor. Factor can be a float
    :param value: float
    :param factor: float
    :return: float
    """

    return factor * round(value / factor)


def get_annotations_from_default_boxes(default_boxes_matrix):
    """
    Converts a matrix of default boxes into a list of Annotation instances
    :param default_boxes_matrix: 2D numpy array with 5 elements per row -
    [x_left, y_top, x_right, y_bottom, category_id]
    :return: list of Annotation instances
    """

    integer_boxes_matrix = default_boxes_matrix.astype(np.int32)
    return [Annotation(default_box[:4]) for default_box in integer_boxes_matrix]


def get_matched_boxes_indices(template_box, boxes_matrix):
    """
    Checks for intersection over union between a matrix of boxes and a template box and returns indices
    of boxes from the matrix that have intersection over union with the template box above 0.5
    :param template_box: bounding box tuple (x_left, y_top, x_right, y_bottom)
    :param boxes_matrix: 2D numpy array of boxes in [x_left, y_top, x_right, y_bottom] order
    :return: 1D array of ints
    """

    template_x_center = (template_box[0] + template_box[2]) / 2

    # Candidate bounding box has to have its left side border to the left of template box's center and
    # right side border to the right of template box's center, otherwise IOU can't be over 0.5
    horizontal_filter = (boxes_matrix[:, 0] < template_x_center) & (boxes_matrix[:, 2] > template_x_center)

    template_y_center = (template_box[1] + template_box[3]) / 2

    # Candidate bounding box has to have its top side border above template box's center and
    # bottom side border below template box's center, otherwise IOU can't be over 0.5
    vertical_filter = (boxes_matrix[:, 1] < template_y_center) & (boxes_matrix[:, 3] > template_y_center)

    filtered_indices = np.where(horizontal_filter & vertical_filter)[0]

    filtered_boxes = boxes_matrix[filtered_indices]

    template_size = (template_box[2] - template_box[0]) * (template_box[3] - template_box[1])
    filtered_boxes_sizes = (filtered_boxes[:, 2] - filtered_boxes[:, 0]) * (filtered_boxes[:, 3] - filtered_boxes[:, 1])

    # Sizes ratio must be within (0.5, 2) range, otherwise IOU can't be above 0.5
    sizes_ratios = filtered_boxes_sizes / template_size
    sizes_filter = (sizes_ratios >= 0.5) & (sizes_ratios <= 2)

    # Filter out indices by size_filter as well
    filtered_indices = filtered_indices[np.where(sizes_filter)[0]]

    # Computer IOUs for boxes that made it through simple filters
    ious = get_vectorized_intersection_over_union(template_box, boxes_matrix[filtered_indices])

    filtered_indices = filtered_indices[np.where(ious > 0.5)[0]]
    return filtered_indices


def get_vectorized_intersection_over_union(template_box, boxes_matrix):
    """
    Computes IOU of template_box with every box in boxes_matrix
    :param template_box: box in corner format
    :param boxes_matrix: 2D numpy array, each row represents a box in corner format
    :return: 1D numpy array of floats representing intersection over union values
    """

    # General idea: get the horizontal intersection, then vertical intersection, multiply results
    # Then do the same for unions. Finally compute intersection / union

    # To get horizontal intersection - get larger of x-left values and smaller of x-right values,
    # then compute max(0, diff)
    larger_x_left_values = np.maximum(template_box[0], boxes_matrix[:, 0])
    smaller_x_right_values = np.minimum(template_box[2], boxes_matrix[:, 2])

    horizontal_intersections = np.maximum(0, smaller_x_right_values - larger_x_left_values)

    # To get vertical intersection - get larger of y-top values and smaller of y-bottom values,
    # then compute max(0, diff)
    larger_y_top_values = np.maximum(template_box[1], boxes_matrix[:, 1])
    smaller_y_bottom_values = np.minimum(template_box[3], boxes_matrix[:, 3])

    vertical_intersections = np.maximum(0, smaller_y_bottom_values - larger_y_top_values)

    intersections = horizontal_intersections * vertical_intersections

    # Union is just sum of areas of two boxes minus their intersectino
    template_box_area = (template_box[2] - template_box[0]) * (template_box[3] - template_box[1])
    boxes_areas = (boxes_matrix[:, 2] - boxes_matrix[:, 0]) * (boxes_matrix[:, 3] - boxes_matrix[:, 1])

    unions = template_box_area + boxes_areas - intersections

    return intersections / unions


def get_objects_sizes_analysis(annotations, size_factor):
    """
    Performs analysis of common objects sizes, rounded to a specified size factor,
    and returns a list of tuples (count, size) in descending order by count
    :param annotations: list of Annotation instances
    :param size_factor: int, factor within a multiple of which we group objects sizes together
    :return: list of tuples (count, size)
    """

    sizes = [annotation.size for annotation in annotations]

    # Within a small margin, force objects to be the same size, so we can see frequent sizes groups more easily
    sizes = [get_target_shape(size, size_factor=size_factor) for size in sizes]

    sizes_counter = collections.Counter(sizes)
    ordered_sizes = sorted(sizes_counter.items(), key=lambda x: x[1], reverse=True)

    return [(count, size) for size, count in ordered_sizes]


def get_objects_aspect_ratios_analysis(annotations, size_factor):
    """
    Performs analysis of aspect rations and
    and returns a list of tuples (count, aspect ratio) in descending order by count
    :param annotations: list of Annotation instances
    :param size_factor: int, factor within a multiple of which we group objects sizes together
    :return: list of tuples (count, aspect ratio)
    """

    sizes = [annotation.size for annotation in annotations]

    # Within a small margin, force objects to be the same size, so we can see frequent sizes groups more easily
    sizes = [get_target_shape(size, size_factor=size_factor) for size in sizes]

    aspect_ratios = [width / height for (height, width) in sizes]
    aspect_ratios = [round_to_factor(aspect_ratio, 0.2) for aspect_ratio in aspect_ratios]

    aspect_ratios_counter = collections.Counter(aspect_ratios)
    return sorted(aspect_ratios_counter.items(), key=lambda x: x[1], reverse=True)


def analyze_annotations(annotations):
    """
    Given list of annotations, performs sizes and aspect ratios analysis
    :param annotations: list of Annotation instances
    """

    counts_sizes_tuples = get_objects_sizes_analysis(annotations, size_factor=5)

    print("\nObjects' sizes")
    for count, size in counts_sizes_tuples[:100]:
        print("{} -> {}".format(count, size))

    ordered_aspect_ratios_counter = get_objects_aspect_ratios_analysis(annotations, size_factor=5)

    print("\nObjects' aspect ratios")
    for aspect_ratio, count in ordered_aspect_ratios_counter[:100]:
        print("{} -> {}".format(count, aspect_ratio))

    annotations_over_large_size = []
    annotations_below_small_size = []

    for count, size_tuple in counts_sizes_tuples:

        if max(size_tuple) > 100:
            annotations_over_large_size.append(count)

        if max(size_tuple) < 50:
            annotations_below_small_size.append(count)

    print("Above large size ratio: {}".format(sum(annotations_over_large_size) / len(annotations)))
    print("Below small size ratio: {}".format(sum(annotations_below_small_size) / len(annotations)))


def get_detections_after_soft_non_maximum_suppression(detections, sigma, score_threshold):
    """
    Soft non-maximum suppression algorithm.
    Implementation adapted from https://github.com/OneDirection9/soft-nms
    Args:
        detections (numpy.array): Detection results with shape `(num, 5)`,
            data in second dimension are [x_min, y_min, x_max, y_max, score] respectively.
        sigma (float): Gaussian function parameter. Only work when method
            is `gaussian`.
        score_threshold (float): Boxes with score below this value will be discarded
    Returns:
        numpy.array: Retained boxes.
    .. _`Improving Object Detection With One Line of Code`:
        https://arxiv.org/abs/1704.04503
    """

    areas = (detections[:, 2] - detections[:, 0] + 1) * (detections[:, 3] - detections[:, 1] + 1)
    # expand detections with areas, so that the second dimension is
    # x_min, y_min, x_max, y_max, score, area
    detections = np.concatenate([detections, areas.reshape(-1, 1)], axis=1)

    retained_detections = []

    while detections.size > 0:

        # Get index for detection with max score, then swap that detection with detection at index 0.
        # This way we will get detection with max score at index 0 in detections array
        max_score_index = np.argmax(detections[:, 4], axis=0)
        detections[[0, max_score_index]] = detections[[max_score_index, 0]]

        # Save max score detection to retained detections
        retained_detections.append(detections[0])

        # Compute intersection over union between top score box and all other boxes
        min_x = np.maximum(detections[0, 0], detections[1:, 0])
        min_y = np.maximum(detections[0, 1], detections[1:, 1])
        max_x = np.minimum(detections[0, 2], detections[1:, 2])
        max_y = np.minimum(detections[0, 3], detections[1:, 3])

        overlap_width = np.maximum(max_x - min_x + 1, 0.0)
        overlap_height = np.maximum(max_y - min_y + 1, 0.0)

        intersection_area = overlap_width * overlap_height
        intersection_over_union = intersection_area / (detections[0, 5] + detections[1:, 5] - intersection_area)

        # Update detections scores for all detections other than max score - we don't want to affect its score.
        # Scores are updated using an exponential function such that detections that have no intersection with top
        # score detection aren't affected, and boxes that have iou of 1 with top score detection have their
        # scores set to zero
        detections[1:, 4] *= np.exp(-(intersection_over_union * intersection_over_union) / sigma)

        # Discard detections with scores below score threshold. Take care to shift indices by +1 to account for fact
        # we are leaving out top score detection at index 0
        retained_detections_indices = np.where(detections[1:, 4] >= score_threshold)[0] + 1
        detections = detections[retained_detections_indices]

    return np.array(retained_detections)

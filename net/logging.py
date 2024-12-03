""""
Module with logging logic
"""

import vlogging

import net.data
import net.plot
import net.ssd
import net.utilities


def log_default_boxes_matches_for_single_sample(
        logger, ssd_input_generator, default_boxes_factory, categories_to_colors_map, font_path):
    """
    Logs default boxes matches for a single sample. If no default box was matched with any annotation,
    nothing is logger.
    :param logger: logger instance
    :param ssd_input_generator: generator that outputs (image, annotations) tuples
    :param default_boxes_factory: net.ssd.DefaultBoxesFactory instance
    :param categories_to_colors_map: dictionary mapping categories labels to colors
    :param font_path: path to font to be used to annotate images
    """

    image, annotations = next(ssd_input_generator)
    default_boxes_matrix = default_boxes_factory.get_default_boxes_matrix(image.shape)

    all_matched_default_boxes_indices = []

    for annotation in annotations:
        matched_default_boxes_indices = net.utilities.get_matched_boxes_indices(
            annotation.bounding_box, default_boxes_matrix, threshold=0.5)

        all_matched_default_boxes_indices.extend(matched_default_boxes_indices.tolist())

    if len(all_matched_default_boxes_indices) > 0:
        annotations_colors = [categories_to_colors_map[annotation.label] for annotation in annotations]

        annotated_image = net.plot.get_annotated_image(
            image=net.data.ImageProcessor.get_denormalized_image(image),
            annotations=annotations,
            colors=annotations_colors,
            draw_labels=True,
            font_path=font_path)

        matched_boxes = default_boxes_matrix[all_matched_default_boxes_indices]
        matched_boxes_image = net.plot.get_image_with_boxes(
            image=net.data.ImageProcessor.get_denormalized_image(image),
            boxes=matched_boxes,
            color=(0, 255, 0))

        logger.info(vlogging.VisualRecord("Default boxes", [annotated_image, matched_boxes_image]))


def log_single_prediction(logger, model, default_boxes_factory, samples_iterator, config):
    """
    Log network prediction on a single sample. Draws a single sample from samples_iterator
    :param logger: logger instance
    :param model: net.ml.VGGishModel instance
    :param default_boxes_factory: net.ssd.DefaultBoxesFactory instance
    :param samples_iterator: iterator that yields (image, ground truth annotations) tuples
    :param config: dictionary with configuration options
    """

    image, ground_truth_annotations = next(samples_iterator)

    ground_truth_annotations_image = net.plot.get_annotated_image(
        image=net.data.ImageProcessor.get_denormalized_image(image),
        annotations=ground_truth_annotations,
        colors=[(255, 0, 0)] * len(ground_truth_annotations),
        font_path=config["font_path"])

    softmax_predictions_matrix, offsets_predictions_matrix = model.predict(image)

    default_boxes_matrix = default_boxes_factory.get_default_boxes_matrix(image.shape)

    predictions_with_nms = net.ssd.PredictionsComputer(
        categories=config["categories"],
        threshold=0.5,
        use_non_maximum_suppression=True).get_predictions(
            bounding_boxes_matrix=default_boxes_matrix + offsets_predictions_matrix,
            softmax_predictions_matrix=softmax_predictions_matrix)

    predicted_annotations_image_with_nms = net.plot.get_annotated_image(
        image=net.data.ImageProcessor.get_denormalized_image(image),
        annotations=predictions_with_nms,
        colors=[(0, 255, 0)] * len(predictions_with_nms),
        font_path=config["font_path"])

    logger.info(vlogging.VisualRecord(
        "Ground truth vs predictions vs predictions with nms",
        [ground_truth_annotations_image, predicted_annotations_image_with_nms]))


def get_single_sample_debugging_visual_record(
        image, ground_truth_annotations, matched_default_boxes, predicted_annotations, config):
    """
    Get debugging visual record for a single sample
    :param image: 3D numpy array, image
    :param ground_truth_annotations: list of ground truth Annotation instances
    :param matched_default_boxes: 2D numpy array of default boxes matched with ground truth annotations
    :param predicted_annotations: list of predicted Annotation instances
    :param config: dictionary with configuration parameters
    :return: vlogging.VisualRecord instance
    """

    ground_truth_annotations_image = net.plot.get_annotated_image(
        image=image,
        annotations=ground_truth_annotations,
        colors=[(255, 0, 0)] * len(ground_truth_annotations),
        font_path=config["font_path"])

    matched_boxes_image = net.plot.get_image_with_boxes(
        image=image,
        boxes=matched_default_boxes,
        color=(0, 255, 0))

    predicted_annotations_image = net.plot.get_annotated_image(
        image=image,
        annotations=predicted_annotations,
        colors=[(0, 255, 0)] * len(predicted_annotations),
        draw_labels=True,
        font_path=config["font_path"])

    message = "Ground truth annotations count: {}, matched default boxes count: {}, predictions count: {}".format(
        len(ground_truth_annotations), len(matched_default_boxes), len(predicted_annotations))

    record = vlogging.VisualRecord(
        title="Debug info - raw image, ground truth annotations, matched default boxes, predictions",
        imgs=[
            image,
            ground_truth_annotations_image,
            matched_boxes_image,
            predicted_annotations_image
        ],
        footnotes=message)

    return record


def log_single_sample_debugging_info(
        logger, model, default_boxes_factory, samples_iterator, config):
    """
    Log debugging info for a single sample. Draws a single sample from samples_iterator
    :param logger: logger instance
    :param model: net.ml.VGGishModel instance
    :param default_boxes_factory: net.ssd.DefaultBoxesFactory instance
    :param samples_iterator: iterator that yields (image, ground truth annotations) tuples
    :param config: dictionary with configuration options
    """

    image, ground_truth_annotations = next(samples_iterator)

    default_boxes_matrix = default_boxes_factory.get_default_boxes_matrix(image.shape)

    matched_default_boxes = net.ssd.get_matched_default_boxes(
        annotations=ground_truth_annotations,
        default_boxes_matrix=default_boxes_matrix)

    # Get predictions
    softmax_predictions_matrix, offsets_predictions_matrix = model.predict(image)

    # Get annotations boxes and labels from predictions matrix and default boxes matrix
    predictions = net.ssd.PredictionsComputer(
        categories=config["categories"],
        threshold=0.5,
        use_non_maximum_suppression=False).get_predictions(
            bounding_boxes_matrix=default_boxes_matrix + offsets_predictions_matrix,
            softmax_predictions_matrix=softmax_predictions_matrix)

    record = get_single_sample_debugging_visual_record(
        image=net.data.ImageProcessor.get_denormalized_image(image),
        ground_truth_annotations=ground_truth_annotations,
        matched_default_boxes=matched_default_boxes,
        predicted_annotations=predictions,
        config=config)

    logger.info(record)

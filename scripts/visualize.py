"""
Script with visualizations of data generators outputs, model prediction, etc
"""

import argparse
import sys

import numpy as np
import tensorflow as tf
import tqdm
import vlogging
import yaml

import net.utilities
import net.data
import net.ml
import net.plot
import net.ssd


def log_voc_samples_generator_output(logger, config):
    """
    Logs voc samples generator output
    """

    samples_loader = net.data.VOCSamplesDataLoader(
        data_directory=config["voc"]["data_directory"],
        data_set_path=config["voc"]["validation_set_path"],
        categories=config["categories"],
        size_factor=config["size_factor"],
        augmentation_pipeline=net.data.get_image_augmentation_pipeline())

    generator = iter(samples_loader)

    categories_to_colors_map = net.utilities.get_categories_to_colors_map(config["categories"])

    for _ in tqdm.tqdm(range(100)):

        image, annotations = next(generator)

        colors = [categories_to_colors_map[annotation.label] for annotation in annotations]

        image = net.plot.get_annotated_image(
            image=net.data.ImageProcessor.get_denormalized_image(image),
            annotations=annotations,
            colors=colors,
            draw_labels=True,
            font_path=config["font_path"])

        labels = [annotation.label for annotation in annotations]
        message = "{} - {}".format(image.shape[:2], labels)

        logger.info(vlogging.VisualRecord("Data", [image], message))


def log_sample_with_odd_sized_annotation(logger, image, annotations, categories_to_colors_map, font_path):
    """
    Logs into a logger a single image and its annotations, writing dimensions details of each annotation
    :param logger: logger instance
    :param image: image to log
    :param annotations: annotations for the image to log
    :param categories_to_colors_map: dictionary mapping categories labels to colors
    :param font_path: path to font to be used to draw labels text on image
    """

    colors = [categories_to_colors_map[annotation.label] for annotation in annotations]

    image = net.plot.get_annotated_image(
        image=image,
        annotations=annotations,
        colors=colors,
        draw_labels=True,
        font_path=font_path)

    labels = []

    for annotation in annotations:
        label = "{} - width: {}, height: {}, aspect ratio: {}".format(
            annotation.label, annotation.width, annotation.height, annotation.aspect_ratio)

        labels.append(label)

    labels_message = "\n".join(labels)
    message = "{}\n{}".format(image.shape[:2], labels_message)

    logger.info(vlogging.VisualRecord("Data", [image], message))


def log_samples_with_odd_sized_annotations(logger, config):
    """
    Logs images with objects of odd sizes - suspiciously small, strange aspect ratios, etc
    """

    samples_loader = net.data.VOCSamplesDataLoader(
        data_directory=config["voc"]["data_directory"],
        data_set_path=config["voc"]["validation_set_path"],
        categories=config["categories"],
        size_factor=config["size_factor"])

    generator = iter(samples_loader)

    categories_to_colors_map = net.utilities.get_categories_to_colors_map(config["categories"])

    all_annotations_count = 0
    unusual_sized_annotation_count = 0

    for _ in tqdm.tqdm(range(len(samples_loader))):

        image, annotations = next(generator)

        all_annotations_count += len(annotations)

        unusual_sized_annotations = []

        for annotation in annotations:

            if net.utilities.is_annotation_size_unusual(
                    annotation,
                    config["objects_filtering"]["minimum_size"],
                    config["objects_filtering"]["minimum_aspect_ratio"],
                    config["objects_filtering"]["maximum_aspect_ratio"]):

                unusual_sized_annotations.append(annotation)

        unusual_sized_annotation_count += len(unusual_sized_annotations)

        if len(unusual_sized_annotations) > 0:

            log_sample_with_odd_sized_annotation(
                logger=logger,
                image=net.data.ImageProcessor.get_denormalized_image(image),
                annotations=unusual_sized_annotations,
                categories_to_colors_map=categories_to_colors_map,
                font_path=config["font_path"])

    print("Unusual object sizes counts to objects count: {}/{}".format(
        unusual_sized_annotation_count, all_annotations_count))


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
            annotation.bounding_box, default_boxes_matrix)

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


def get_default_boxes_matches_image(image, annotations, default_boxes_matrix):
    """
    Get image with default boxes matched to annotations
    :param image: 3D numpy array
    :param annotations: list of Annotation instances
    :param default_boxes_matrix: 2D array of default boxes
    :return: 3D numpy array
    """

    # Get default boxes matches
    all_matched_default_boxes_indices = []

    for annotation in annotations:

        matched_default_boxes_indices = net.utilities.get_matched_boxes_indices(
            annotation.bounding_box, default_boxes_matrix)

        all_matched_default_boxes_indices.extend(matched_default_boxes_indices.tolist())

    matched_boxes = default_boxes_matrix[all_matched_default_boxes_indices]

    matched_boxes_image = net.plot.get_image_with_boxes(
        image=image,
        boxes=matched_boxes,
        color=(0, 255, 0))

    return matched_boxes_image


def log_default_boxes_matches(logger, config):
    """
    Log default boxes matches
    """

    samples_loader = net.data.VOCSamplesDataLoader(
        data_directory=config["voc"]["data_directory"],
        data_set_path=config["voc"]["validation_set_path"],
        categories=config["categories"],
        size_factor=config["size_factor"])

    default_boxes_factory = net.ssd.DefaultBoxesFactory(config["vggish_model_configuration"])
    categories_to_colors_map = net.utilities.get_categories_to_colors_map(config["categories"])

    for _ in tqdm.tqdm(range(100)):

        log_default_boxes_matches_for_single_sample(
            logger, iter(samples_loader), default_boxes_factory, categories_to_colors_map, config["font_path"])


def log_ssd_training_loop_data_loader_outputs(logger, config):
    """
    Logger function for visually confirming that net.ssd.SSDTrainingLoopDataLoader outputs correct results
    """

    samples_loader = net.data.VOCSamplesDataLoader(
        data_directory=config["voc"]["data_directory"],
        data_set_path=config["voc"]["validation_set_path"],
        categories=config["categories"],
        size_factor=config["size_factor"])

    ssd_validation_samples_loader = net.ssd.SSDTrainingLoopDataLoader(
        voc_samples_data_loader=samples_loader,
        ssd_model_configuration=config["vggish_model_configuration"])

    default_boxes_factory = net.ssd.DefaultBoxesFactory(config["vggish_model_configuration"])

    iterator = iter(ssd_validation_samples_loader)

    for _ in range(20):

        image, default_boxes_categories_ids_vector = next(iterator)

        default_boxes_matrix = default_boxes_factory.get_default_boxes_matrix(image.shape)

        matched_boxes = default_boxes_matrix[default_boxes_categories_ids_vector > 0]
        matched_boxes_image = net.plot.get_image_with_boxes(
            image=net.data.ImageProcessor.get_denormalized_image(image),
            boxes=matched_boxes,
            color=(0, 255, 0))

        logger.info(vlogging.VisualRecord(
            "image and default boxes matches",
            [net.data.ImageProcessor.get_denormalized_image(image), matched_boxes_image]))


def log_single_prediction(logger, network, session, default_boxes_factory, samples_iterator, config):
    """
    Log network prediction on a single sample. Draws a single sample from samples_iterator
    :param logger: logger instance
    :param network: net.ml.VGGishNetwork instance
    :param session: tensorflow session
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

    softmax_predictions_matrix = session.run(
        network.batch_of_softmax_predictions_matrices_op,
        feed_dict={network.input_placeholder: np.array([image])})[0]

    default_boxes_matrix = default_boxes_factory.get_default_boxes_matrix(image.shape)

    # Get annotations boxes and labels from predictions matrix and default boxes matrix
    predicted_annotations = net.ssd.get_predicted_annotations(
        default_boxes_matrix=default_boxes_matrix,
        softmax_predictions_matrix=softmax_predictions_matrix,
        categories=config["categories"],
        threshold=0.5)

    predicted_annotations_image = net.plot.get_annotated_image(
        image=net.data.ImageProcessor.get_denormalized_image(image),
        annotations=predicted_annotations,
        colors=[(0, 255, 0)] * len(predicted_annotations),
        font_path=config["font_path"])

    logger.info(vlogging.VisualRecord(
        "Ground truth vs predictions", [ground_truth_annotations_image, predicted_annotations_image]))


def log_predictions(logger, config):
    """
    Log network predictions
    :param logger: logger instance
    :param config: dictionary with configuration options
    """

    network = net.ml.VGGishNetwork(
        model_configuration=config["vggish_model_configuration"],
        categories_count=len(config["categories"]))

    session = tf.keras.backend.get_session()

    model = net.ml.VGGishModel(session, network)
    model.load(config["model_checkpoint_path"])

    validation_samples_loader = net.data.VOCSamplesDataLoader(
        data_directory=config["voc"]["data_directory"],
        data_set_path=config["voc"]["validation_set_path"],
        categories=config["categories"],
        size_factor=config["size_factor"])

    default_boxes_factory = net.ssd.DefaultBoxesFactory(model_configuration=config["vggish_model_configuration"])
    iterator = iter(validation_samples_loader)

    for _ in tqdm.tqdm(range(40)):

        log_single_prediction(
            logger=logger,
            network=network,
            session=session,
            default_boxes_factory=default_boxes_factory,
            samples_iterator=iterator,
            config=config)


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
        title="Debug info",
        imgs=[
            image,
            ground_truth_annotations_image,
            matched_boxes_image,
            predicted_annotations_image
        ],
        footnotes=message)

    return record


def log_single_sample_debugging_info(
        logger, network, session, default_boxes_factory, samples_iterator, config):
    """
    Log debugging info for a single sample. Draws a single sample from samples_iterator
    :param logger: logger instance
    :param network: net.ml.VGGishNetwork instance
    :param session: tensorflow Session instance
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
    softmax_predictions_matrix = session.run(
        network.batch_of_softmax_predictions_matrices_op,
        feed_dict={network.input_placeholder: np.array([image])})[0]

    # Get annotations boxes and labels from predictions matrix and default boxes matrix
    predicted_annotations = net.ssd.get_predicted_annotations(
        default_boxes_matrix=default_boxes_matrix,
        softmax_predictions_matrix=softmax_predictions_matrix,
        categories=config["categories"],
        threshold=0.5)

    record = get_single_sample_debugging_visual_record(
        image=net.data.ImageProcessor.get_denormalized_image(image),
        ground_truth_annotations=ground_truth_annotations,
        matched_default_boxes=matched_default_boxes,
        predicted_annotations=predicted_annotations,
        config=config)

    logger.info(record)


def log_debugging_info(logger, config):
    """
    Log debug info
    :param logger: logger instance
    :param config: dictionary with configuration options
    """

    network = net.ml.VGGishNetwork(
        model_configuration=config["vggish_model_configuration"],
        categories_count=len(config["categories"]))

    session = tf.keras.backend.get_session()

    model = net.ml.VGGishModel(session, network)
    model.load(config["model_checkpoint_path"])

    validation_samples_loader = net.data.VOCSamplesDataLoader(
        data_directory=config["voc"]["data_directory"],
        data_set_path=config["voc"]["train_set_path"],
        categories=config["categories"],
        size_factor=config["size_factor"])

    default_boxes_factory = net.ssd.DefaultBoxesFactory(model_configuration=config["vggish_model_configuration"])
    iterator = iter(validation_samples_loader)

    for _ in tqdm.tqdm(range(20)):

        log_single_sample_debugging_info(
            logger=logger,
            network=network,
            session=session,
            default_boxes_factory=default_boxes_factory,
            samples_iterator=iterator,
            config=config)


def main():
    """
    Script entry point
    """

    parser = argparse.ArgumentParser()

    parser.add_argument('--config', action="store", required=True)
    arguments = parser.parse_args(sys.argv[1:])

    with open(arguments.config) as file:
        config = yaml.safe_load(file)

    logger = net.utilities.get_logger(config["log_path"])

    # log_voc_samples_generator_output(logger, config)
    # log_samples_with_odd_sized_annotations(logger, config)

    # log_default_boxes_matches(logger, config)

    # log_ssd_training_loop_data_loader_outputs(logger, config)
    # log_predictions(logger, config)
    log_debugging_info(logger, config)


if __name__ == "__main__":

    main()

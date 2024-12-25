"""
Module with visualization commands
"""

import invoke


@invoke.task
def log_voc_samples_generator_output(_context, config_path):
    """
    Logs voc samples generator output

    Args:
        _context (invoke.Context): context instance
        config_path (str): path to configuration file
    """

    import tqdm
    import vlogging
    import yaml

    import net.data
    import net.plot
    import net.utilities

    with open(config_path, encoding="utf-8") as file:
        config = yaml.safe_load(file)

    logger = net.utilities.get_logger(config["log_path"])

    generator = iter(net.data.VOCSamplesDataLoader(
        data_directory=config["voc"]["data_directory"],
        data_set_path=config["voc"]["validation_set_path"],
        categories=config["categories"],
        size_factor=config["size_factor"],
        augmentation_pipeline=net.data.get_image_augmentation_pipeline()))

    categories_to_colors_map = net.utilities.get_categories_to_colors_map(config["categories"])

    for _ in tqdm.tqdm(range(40)):

        image, annotations = next(generator)

        image = net.plot.get_annotated_image(
            image=net.data.ImageProcessor.get_denormalized_image(image),
            annotations=annotations,
            colors=[categories_to_colors_map[annotation.label] for annotation in annotations],
            draw_labels=True,
            font_path=config["font_path"])

        labels = [annotation.label for annotation in annotations]
        message = "{} - {}".format(image.shape[:2], labels)

        logger.info(vlogging.VisualRecord("Data", [image], message))


@invoke.task
def log_default_boxes_matches(_context, config_path):
    """
    Logs default boxes matches

    Args:
        _context (invoke.Context): context instance
        config_path (str): path to configuration file
    """

    import tqdm
    import yaml

    import net.data
    import net.logging
    import net.ssd
    import net.utilities

    with open(config_path, encoding="utf-8") as file:
        config = yaml.safe_load(file)

    logger = net.utilities.get_logger(config["log_path"])

    samples_loader = net.data.VOCSamplesDataLoader(
        data_directory=config["voc"]["data_directory"],
        data_set_path=config["voc"]["validation_set_path"],
        categories=config["categories"],
        size_factor=config["size_factor"])

    default_boxes_factory = net.ssd.DefaultBoxesFactory(config["vggish_model_configuration"])
    categories_to_colors_map = net.utilities.get_categories_to_colors_map(config["categories"])

    for _ in tqdm.tqdm(range(40)):

        net.logging.log_default_boxes_matches_for_single_sample(
            logger, iter(samples_loader), default_boxes_factory, categories_to_colors_map, config["font_path"])


@invoke.task
def log_ssd_training_loop_data_loader_outputs(_context, config_path):
    """
    Command visually confirming that net.ssd.SSDTrainingLoopDataLoader outputs correct results

    Args:
        _context (invoke.Context): context instance
        config_path (str): path to configuration file
    """

    import tqdm
    import yaml
    import vlogging

    import net.data
    import net.plot
    import net.ssd
    import net.utilities

    with open(config_path, encoding="utf-8") as file:
        config = yaml.safe_load(file)

    logger = net.utilities.get_logger(config["log_path"])

    iterator = iter(net.ssd.SSDTrainingLoopDataLoader(
        voc_samples_data_loader=net.data.VOCSamplesDataLoader(
            data_directory=config["voc"]["data_directory"],
            data_set_path=config["voc"]["validation_set_path"],
            categories=config["categories"],
            size_factor=config["size_factor"]),
        ssd_model_configuration=config["vggish_model_configuration"]))

    default_boxes_factory = net.ssd.DefaultBoxesFactory(config["vggish_model_configuration"])

    for _ in tqdm.tqdm(range(20)):

        image, default_boxes_categories_ids_vector, _, _ = next(iterator)

        default_boxes_matrix = default_boxes_factory.get_default_boxes_matrix(image.shape)

        matched_boxes = default_boxes_matrix[default_boxes_categories_ids_vector > 0]

        matched_boxes_image = net.plot.get_image_with_boxes(
            image=net.data.ImageProcessor.get_denormalized_image(image),
            boxes=matched_boxes,
            color=(0, 255, 0))

        logger.info(vlogging.VisualRecord(
            "image and default boxes matches",
            [net.data.ImageProcessor.get_denormalized_image(image), matched_boxes_image]))


@invoke.task
def log_predictions(_context, config_path):
    """
    Log predictions

    Args:
        _context (invoke.Context): context instance
        config_path (str): path to configuration file
    """

    import box
    import tqdm
    import yaml

    import net.data
    import net.logging
    import net.ssd
    import net.ml
    import net.utilities

    with open(config_path, encoding="utf-8") as file:
        config = box.Box(yaml.safe_load(file))

    logger = net.utilities.get_logger(config["log_path"])

    network = net.ml.VGGishNetwork(
        model_configuration=config["vggish_model_configuration"],
        categories_count=len(config["categories"]))

    network.model.load_weights(config["model_checkpoint_path"])

    validation_samples_loader = net.data.VOCSamplesDataLoader(
        data_directory=config["voc"]["data_directory"],
        data_set_path=config["voc"]["validation_set_path"],
        categories=config["categories"],
        size_factor=config["size_factor"])

    default_boxes_factory = net.ssd.DefaultBoxesFactory(model_configuration=config["vggish_model_configuration"])
    iterator = iter(validation_samples_loader)

    for _ in tqdm.tqdm(range(40)):

        net.logging.log_single_prediction(
            logger=logger,
            model=network,
            default_boxes_factory=default_boxes_factory,
            samples_iterator=iterator,
            config=config)


@invoke.task
def log_debugging_info(_context, config_path):
    """
    Log debug info

    Args:
        _context (invoke.Context): context instance
        config_path (str): path to configuration file
    """

    import tqdm
    import yaml

    import net.data
    import net.logging
    import net.ssd
    import net.ml
    import net.utilities

    with open(config_path, encoding="utf-8") as file:
        config = yaml.safe_load(file)

    logger = net.utilities.get_logger(config["log_path"])

    network = net.ml.VGGishNetwork(
        model_configuration=config["vggish_model_configuration"],
        categories_count=len(config["categories"]))

    network.model.load_weights(config["model_checkpoint_path"])

    iterator = iter(net.data.VOCSamplesDataLoader(
        data_directory=config["voc"]["data_directory"],
        data_set_path=config["voc"]["train_set_path"],
        categories=config["categories"],
        size_factor=config["size_factor"]))

    default_boxes_factory = net.ssd.DefaultBoxesFactory(
        model_configuration=config["vggish_model_configuration"])

    for _ in tqdm.tqdm(range(20)):

        net.logging.log_single_sample_debugging_info(
            logger=logger,
            model=network,
            default_boxes_factory=default_boxes_factory,
            samples_iterator=iterator,
            config=config)


@invoke.task
def log_augmentations(_context, config_path):
    """
    Log augmentations

    Args:
        _context (invoke.Context): context instance
        config_path (str): path to configuration file
    """

    import tqdm
    import vlogging
    import yaml

    import net.data
    import net.plot
    import net.utilities

    with open(config_path, encoding="utf-8") as file:
        config = yaml.safe_load(file)

    logger = net.utilities.get_logger(config["log_path"])

    iterator = iter(net.data.VOCSamplesDataLoader(
        data_directory=config["voc"]["data_directory"],
        data_set_path=config["voc"]["train_set_path"],
        categories=config["categories"],
        size_factor=config["size_factor"]))

    augmentation_pipeline = net.data.get_image_augmentation_pipeline()

    for _ in tqdm.tqdm(range(10)):

        image, annotations = next(iterator)
        image = net.data.ImageProcessor.get_denormalized_image(image)

        augmented_image, augmented_annotations = net.data.get_augmented_sample(
            image=image, annotations=annotations, augmentation_pipeline=augmentation_pipeline)

        logger.info(
            vlogging.VisualRecord(
                "sample",
                imgs=[
                    net.plot.get_annotated_image(
                        image=image,
                        annotations=annotations,
                        colors=[(255, 0, 0)] * len(annotations),
                        draw_labels=False),
                    net.plot.get_annotated_image(
                        image=augmented_image,
                        annotations=augmented_annotations,
                        colors=[(255, 0, 0)] * len(augmented_annotations),
                        draw_labels=False)
                ]
            )
        )

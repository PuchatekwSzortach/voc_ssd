"""

Module with analysis commands
"""

import invoke


@invoke.task
def analyze_data(_context, config_path):
    """
    Analyze data

    Args:
        _context (invoke.Context): context instance
        config_path (str): path to config file
    """

    import yaml

    import net.analysis
    import net.utilities

    with open(config_path, encoding="utf-8") as file:
        config = yaml.safe_load(file)

    annotations = net.analysis.get_filtered_dataset_annotations(config)
    net.utilities.analyze_annotations(annotations)

    print("Total annotations: {}".format(len(annotations)))


@invoke.task
def analyze_network_theoretical_bounds(_context, config_path):
    """
    Command for analyzing theoretical bounds on model's recall

    Args:
        _context (invoke.Context): context instance
        config_path (str): path to config file
    """

    import tqdm
    import yaml

    import net.data
    import net.ssd
    import net.utilities

    with open(config_path, encoding="utf-8") as file:
        config = yaml.safe_load(file)

    ssd_model_configuration = config["vggish_model_configuration"]

    voc_samples_loader = net.data.VOCSamplesDataLoader(
        data_directory=config["voc"]["data_directory"],
        data_set_path=config["voc"]["validation_set_path"],
        categories=config["categories"],
        size_factor=config["size_factor"])

    matching_analysis_generator = net.ssd.get_matching_analysis_generator(
        ssd_model_configuration=ssd_model_configuration,
        ssd_input_generator=iter(voc_samples_loader),
        threshold=0.5
    )

    matched_annotations = []
    unmatched_annotations = []

    for _ in tqdm.tqdm(range(len(voc_samples_loader))):

        single_image_matched_annotations, single_image_unmatched_annotations = next(matching_analysis_generator)

        matched_annotations.extend(single_image_matched_annotations)
        unmatched_annotations.extend(single_image_unmatched_annotations)

    theoretical_recall = len(matched_annotations) / (len(matched_annotations) + len(unmatched_annotations))

    print("Theoretical recall: {}".format(theoretical_recall))

    # Analyze failures
    net.utilities.analyze_annotations(unmatched_annotations)


@invoke.task
def analyze_network_configuration(_context, config_path):
    """
    Analyze model's configuration, or IOUs between different bounding boxes defined by it

    Args:
        _context (invoke.Context): context instance
        config_path (str): path to config file
    """

    import pprint

    import yaml

    import net.utilities

    with open(config_path, encoding="utf-8") as file:
        config = yaml.safe_load(file)

    model_configuration = config["vggish_model_configuration"]

    for prediction_layer in model_configuration["prediction_heads_order"]:

        layer_configuration = model_configuration[prediction_layer]

        for base_size in layer_configuration["base_bounding_box_sizes"]:

            # Vertical boxes
            for aspect_ratio in layer_configuration["aspect_ratios"]:

                width = aspect_ratio * base_size
                height = base_size

                box_definition = net.utilities.DefaultBoxDefinition(
                    width=width, height=height, step=layer_configuration["image_downscale_factor"])

                overlaps = box_definition.get_overlaps(box_definition)

                pprint.pprint(box_definition)
                pprint.pprint(overlaps)
                print()

            # Horizontal boxes
            for aspect_ratio in layer_configuration["aspect_ratios"]:

                width = base_size
                height = aspect_ratio * base_size

                box_definition = net.utilities.DefaultBoxDefinition(
                    width=width, height=height, step=layer_configuration["image_downscale_factor"])

                overlaps = box_definition.get_overlaps(box_definition)

                pprint.pprint(box_definition)
                pprint.pprint(overlaps)
                print()


@invoke.task
def analyze_objects_detections_predictions(_context, config_path):
    """
    Task to analyze objects detections predictions

    Args:
        _context (invoke.Context): context instance
        config_path (str): path to configurtion file
    """

    import box
    import yaml

    import net.analysis
    import net.data
    import net.ml
    import net.ssd
    import net.utilities

    with open(config_path, encoding="utf-8") as file:
        config = box.Box(yaml.safe_load(file))

        ssd_model_configuration = config["vggish_model_configuration"]

    network = net.ml.VGGishNetwork(
        model_configuration=ssd_model_configuration,
        categories_count=len(config["categories"]))

    network.model.load_weights(config["model_checkpoint_path"])

    validation_samples_loader = net.data.VOCSamplesDataLoader(
        data_directory=config["voc"]["data_directory"],
        data_set_path=config["voc"]["validation_set_path"],
        categories=config["categories"],
        size_factor=config["size_factor"])

    logger = net.utilities.get_logger(config["log_path"])

    default_boxes_factory = net.ssd.DefaultBoxesFactory(model_configuration=ssd_model_configuration)

    thresholds_matching_data_map = net.analysis.MatchingDataComputer(
        samples_loader=validation_samples_loader,
        model=network,
        default_boxes_factory=default_boxes_factory,
        confidence_thresholds=[0, 0.5, 0.9],
        categories=config["categories"],
        post_processing_config=config.post_processing).get_thresholds_matched_data_map()

    net.analysis.log_precision_recall_analysis(
        logger=logger,
        thresholds_matching_data_map=thresholds_matching_data_map)

    net.analysis.log_mean_average_precision_analysis(
        logger=logger,
        thresholds_matching_data_map=thresholds_matching_data_map)

    net.analysis.log_performance_with_annotations_size_analysis(
        logger=logger,
        thresholds_matching_data_map=thresholds_matching_data_map)

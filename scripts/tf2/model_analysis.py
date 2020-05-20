"""
Script with model analysis
"""

import argparse
import sys

import yaml

import net.analysis
import net.data
import net.ml
import net.ssd
import net.tf2
import net.utilities


def main():
    """
    Script entry point
    """

    parser = argparse.ArgumentParser()

    parser.add_argument('--config', action="store", required=True)
    arguments = parser.parse_args(sys.argv[1:])

    with open(arguments.config) as file:
        config = yaml.safe_load(file)

    ssd_model_configuration = config["vggish_model_configuration"]

    network = net.tf2.VGGishNetwork(
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
        thresholds=[0, 0.5, 0.9],
        categories=config["categories"]).get_thresholds_matched_data_map()

    net.analysis.log_precision_recall_analysis(
        logger=logger,
        thresholds_matching_data_map=thresholds_matching_data_map)

    net.analysis.log_mean_average_precision_analysis(
        logger=logger,
        thresholds_matching_data_map=thresholds_matching_data_map)

    # losses_map = net.analysis.get_mean_losses(
    #     model=network,
    #     ssd_model_configuration=ssd_model_configuration,
    #     samples_loader=validation_samples_loader)

    # logger.info("<br><h2>Losses map: {}</h2><br>".format(losses_map))

    net.analysis.log_performance_with_annotations_size_analysis(
        logger=logger,
        thresholds_matching_data_map=thresholds_matching_data_map)


if __name__ == "__main__":

    main()

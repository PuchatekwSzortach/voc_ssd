"""
Script for analyzing theoretical bounds on model's recall
"""

import argparse
import sys

import yaml
import tqdm

import net.data
import net.ssd
import net.utilities


def analyse_theoretical_performance(config):
    """
    Analyse theoretical performance of SSD model on VOC dataset
    """

    voc_samples_loader = net.data.VOCSamplesDataLoader(
        config["voc"]["data_directory"], config["voc"]["validation_set_path"], config["size_factor"])

    ssd_moden_input_data_loader = net.data.SSDModelInputDataLoader(
        voc_samples_loader, config["objects_filtering"])

    matching_analysis_generator = net.ssd.get_matching_analysis_generator(
        config["vggish_model_configuration"], iter(ssd_moden_input_data_loader))

    matched_annotations = []
    unmatched_annotations = []

    for _ in tqdm.tqdm(range(len(ssd_moden_input_data_loader))):

        single_image_matched_annotations, single_image_unmatched_annotations = next(matching_analysis_generator)

        matched_annotations.extend(single_image_matched_annotations)
        unmatched_annotations.extend(single_image_unmatched_annotations)

    ssd_moden_input_data_loader.stop_generator()

    theoretical_recall = len(matched_annotations) / (len(matched_annotations) + len(unmatched_annotations))

    print("Theoretical recall: {}".format(theoretical_recall))

    # Analyze failures
    net.utilities.analyze_annotations(unmatched_annotations)


def main():
    """
    Script entry point
    """

    parser = argparse.ArgumentParser()

    parser.add_argument('--config', action="store", required=True)
    arguments = parser.parse_args(sys.argv[1:])

    with open(arguments.config) as file:
        config = yaml.safe_load(file)

    analyse_theoretical_performance(config)


if __name__ == "__main__":

    main()

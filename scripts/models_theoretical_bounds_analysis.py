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


def analyse_theoretical_precision(config):
    """
    Analyse theoretical precision of SSD model on VOC dataset
    """

    voc_samples_generator_factory = net.data.VOCSamplesGeneratorFactory(
        config["voc"]["data_directory"], config["voc"]["validation_set_path"], config["size_factor"])

    ssd_input_generator_factory = net.data.SSDInputGeneratorFactory(
        voc_samples_generator_factory, config["objects_filtering"])

    ssd_input_generator = ssd_input_generator_factory.get_generator()

    matching_analysis_generator = net.ssd.get_matching_analysis_generator(
        config["vggish_model_configuration"], ssd_input_generator)

    matched_annotations = []
    unmatched_annotations = []

    for _ in tqdm.tqdm(range(10)):

        single_image_matched_annotations, single_image_unmatched_annotations = next(matching_analysis_generator)

        matched_annotations.extend(single_image_matched_annotations)
        unmatched_annotations.extend(single_image_unmatched_annotations)

    ssd_input_generator_factory.stop_generator()

    print(len(matched_annotations))
    print(len(unmatched_annotations))


def main():
    """
    Script entry point
    """

    parser = argparse.ArgumentParser()

    parser.add_argument('--config', action="store", required=True)
    arguments = parser.parse_args(sys.argv[1:])

    with open(arguments.config) as file:
        config = yaml.safe_load(file)

    analyse_theoretical_precision(config)


if __name__ == "__main__":

    main()

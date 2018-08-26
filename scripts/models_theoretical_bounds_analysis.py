"""
Script for analyzing theoretical bounds on model's recall
"""

import argparse
import sys

import yaml
import tqdm

import net.data
import net.ssd


def main():
    """
    Script entry point
    """

    parser = argparse.ArgumentParser()

    parser.add_argument('--config', action="store", required=True)
    arguments = parser.parse_args(sys.argv[1:])

    with open(arguments.config) as file:
        config = yaml.safe_load(file)

    voc_samples_generator_factory = net.data.VOCSamplesGeneratorFactory(
        config["voc"]["data_directory"], config["voc"]["validation_set_path"], config["size_factor"])

    ssd_input_generator_factory = net.data.SSDInputGeneratorFactory(
        voc_samples_generator_factory, config["objects_filtering"])

    ssd_input_generator = ssd_input_generator_factory.get_generator()

    default_boxes_factory = net.ssd.DefaultBoxesFactory(config["vggish_model_configuration"])

    for _ in tqdm.tqdm(range(10)):

        image, annotations = next(ssd_input_generator)
        default_boxes_matrix = default_boxes_factory.get_default_boxes_matrix()

        print(image.shape)
        print(len(annotations))
        print(default_boxes_matrix)

    ssd_input_generator_factory.stop_generator()


if __name__ == "__main__":

    main()

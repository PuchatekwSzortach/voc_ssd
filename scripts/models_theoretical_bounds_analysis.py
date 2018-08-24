"""
Script for analyzing theoretical bounds on model's recall
"""

import argparse
import sys

import yaml
import tqdm

import net.data


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

    generator_factory = net.data.PreprocessedVOCSamplesGeneratorFactory(
        voc_samples_generator_factory, config["objects_filtering"])

    generator = generator_factory.get_generator()

    for _ in tqdm.tqdm(range(10)):

        _ = next(generator)

    generator_factory.stop_generator()


if __name__ == "__main__":

    main()

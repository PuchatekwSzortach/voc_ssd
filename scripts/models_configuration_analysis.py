"""
Script analyzing model's configuration, or IOUs between different bounding boxes defined by it
"""

import argparse
import sys

import yaml

import net.utilities


def analyze_models_configuration(model_configuration):
    """
    Analyzes model's configuration
    :param model_configuration: dictionary defining model's configuration
    """

    for prediction_layer in model_configuration["prediction_heads_order"]:

        layer_configuration = model_configuration[prediction_layer]

        for aspect_ratio in layer_configuration["aspect_ratios"]:

            base_size = layer_configuration["base_bounding_box_size"]

            box_definition = net.utilities.DefaultBoxDefinition(
                width=aspect_ratio * base_size, height=base_size, step=layer_configuration["image_downscale_factor"])

            print(box_definition)


def main():
    """
    Script entry point
    """

    parser = argparse.ArgumentParser()

    parser.add_argument('--config', action="store", required=True)
    arguments = parser.parse_args(sys.argv[1:])

    with open(arguments.config) as file:
        config = yaml.safe_load(file)

    analyze_models_configuration(config["vggish_model_configuration"])


if __name__ == "__main__":
    main()

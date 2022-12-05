"""
Host side code
"""

import omegaconf


def get_config(path: str) -> omegaconf.DictConfig:
    """
    Get config from yaml file. Config is processed with omegaconf

    Args:
        path (str): path to configuration file

    Returns:
        omegaconf.DictConfig: config
    """

    return omegaconf.OmegaConf.load(path)

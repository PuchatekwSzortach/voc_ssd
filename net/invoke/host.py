"""
Host side invoke commands
"""

import invoke


@invoke.task
def build_app_container(context):
    """
    Build app container.

    :param context: invoke.Context instance
    """

    command = (
        "DOCKER_BUILDKIT=1 docker build "
        "--tag net/voc_ssd:latest "
        "-f ./docker/app.Dockerfile ."
    )

    context.run(command, echo=True)


@invoke.task
def run_app_container(context, config_path):
    """
    Run app container

    Args:
        context (invoke.Context): invoke context instance
        config_path (str): path to configuration file
    """

    import os

    import net.host

    config = net.host.get_config(config_path)

    # Define run options that need a bit of computations
    run_options = {
        # Use gpu runtime if host has cuda installed
        "gpu_capabilities": "--gpus all" if "/cuda/" in os.environ["PATH"] else ""
    }

    command = (
        "docker run -it --rm -v $PWD:/app "
        "{gpu_capabilities} "
        f"-v $PWD/{config.common_parameters.base_data_directory_on_host}:/data "
        f"-v {config.common_parameters.logging_output_directory_on_host}:/tmp "
        "net/voc_ssd:latest /bin/bash"
    ).format(**run_options)

    context.run(command, pty=True, echo=True)

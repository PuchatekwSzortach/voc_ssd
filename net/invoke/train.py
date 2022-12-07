"""
Module with machine learning models training code
"""

import invoke


@invoke.task
def train_object_detection_model(_context, config_path):
    """
    Train object detection model

    Args:
        _context (invoke.Context): context instance
        config_path (str): path to configuration file
    """

    import tensorflow as tf
    import yaml

    import net.data
    import net.ssd
    import net.ml

    with open(config_path, encoding="utf-8") as file:
        config = yaml.safe_load(file)

    ssd_model_configuration = config["vggish_model_configuration"]

    training_samples_loader = net.data.VOCSamplesDataLoader(
        data_directory=config["voc"]["data_directory"],
        data_set_path=config["voc"]["train_set_path"],
        categories=config["categories"],
        size_factor=config["size_factor"],
        augmentation_pipeline=net.data.get_image_augmentation_pipeline())

    ssd_training_samples_training_data_loader = net.ssd.SSDTrainingLoopDataLoader(
        voc_samples_data_loader=training_samples_loader,
        ssd_model_configuration=ssd_model_configuration)

    t2_training_samples_loader = net.ml.TF2TrainingLoopDataLoader(
        ssd_training_loop_data_loader=ssd_training_samples_training_data_loader
    )

    validation_samples_loader = net.data.VOCSamplesDataLoader(
        data_directory=config["voc"]["data_directory"],
        data_set_path=config["voc"]["validation_set_path"],
        categories=config["categories"],
        size_factor=config["size_factor"],
        augmentation_pipeline=net.data.get_image_augmentation_pipeline())

    ssd_training_samples_validation_data_loader = net.ssd.SSDTrainingLoopDataLoader(
        voc_samples_data_loader=validation_samples_loader,
        ssd_model_configuration=ssd_model_configuration)

    t2_validation_samples_loader = net.ml.TF2TrainingLoopDataLoader(
        ssd_training_loop_data_loader=ssd_training_samples_validation_data_loader
    )

    network = net.ml.VGGishNetwork(
        model_configuration=ssd_model_configuration,
        categories_count=len(config["categories"]))

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=config["model_checkpoint_path"],
            save_best_only=True,
            save_weights_only=True,
            verbose=1),
        tf.keras.callbacks.EarlyStopping(
            patience=config["train"]["early_stopping_patience"],
            verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(
            patience=config["train"]["reduce_learning_rate_patience"],
            factor=config["train"]["reduce_learning_rate_factor"],
            verbose=1),
        net.ml.HistoryLogger(
            logger=net.utilities.get_logger(config["training_history_log_path"])
        )
    ]

    network.model.fit(
        x=iter(t2_training_samples_loader),
        steps_per_epoch=len(t2_training_samples_loader),
        epochs=100,
        validation_data=iter(t2_validation_samples_loader),
        validation_steps=len(t2_validation_samples_loader),
        callbacks=callbacks
    )

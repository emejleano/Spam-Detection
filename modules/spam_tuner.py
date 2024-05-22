"""
Tuner module
"""

from typing import NamedTuple, Dict, Text, Any
import tensorflow as tf
import tensorflow_transform as tft
import keras_tuner as kt
from keras_tuner.engine import base_tuner
from tfx.components.trainer.fn_args_utils import FnArgs

from tweet_trainer import model_builder, input_fn

# current TrainerFnArgs will be renamed to FnArgs as a util class.
TunerFnResult = NamedTuple('TunerFnResult',
                           [('tuner', base_tuner.BaseTuner),
                            ('fit_kwargs', Dict[Text, Any])])


def tuner_fn(fn_args: FnArgs) -> TunerFnResult:
    """Define hyperparameter tuning function for Keras Tuner.

    Args:
        fn_args (FnArgs): Arguments passed to the function.

    Returns:
        TunerFnResult: NamedTuple containing the Keras Tuner and fit_kwargs.
    """

    # define hyperparameter tuning strategy
    tuner = kt.RandomSearch(model_builder,
                            objective='val_binary_accuracy',
                            max_trials=5,
                            directory=fn_args.working_dir,
                            project_name='kt_random_search')

    # load the training and validation dataset that has been preprocessed
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_graph_path)

    train_set = input_fn(fn_args.train_files, tf_transform_output, 10)
    val_set = input_fn(fn_args.eval_files, tf_transform_output, 10)

    return TunerFnResult(
        tuner=tuner,
        fit_kwargs={
            "callbacks": [
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_binary_accuracy',
                    mode='max',
                    patience=3)],
            'x': train_set,
            'validation_data': val_set,
            'steps_per_epoch': fn_args.train_steps,
            'validation_steps': fn_args.eval_steps,
        })

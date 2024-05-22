# -*- coding: utf-8 -*-
import os
from typing import Text
from absl import logging
from tfx.orchestration import metadata, pipeline
from tfx.orchestration.beam.beam_dag_runner import BeamDagRunner
from tfx.components import CsvExampleGen, StatisticsGen, SchemaGen, ExampleValidator, Transform, Trainer, Tuner, Evaluator, Pusher
from tfx.proto import example_gen_pb2, trainer_pb2, tuner_pb2, pusher_pb2
from tfx.dsl.components.common.resolver import Resolver
from tfx.dsl.input_resolution.strategies.latest_blessed_model_strategy import LatestBlessedModelStrategy
from tfx.types import Channel
from tfx.types.standard_artifacts import Model, ModelBlessing
import tensorflow_model_analysis as tfma

PIPELINE_NAME = "emejleano-pipeline"
OUTPUT_BASE = "outputs"

# Pipeline inputs
DATA_ROOT = "data"
TRANSFORM_MODULE_FILE = "modules/spam_transform.py"
TRAINER_MODULE_FILE = "modules/spam_trainer.py"
TUNER_MODULE_FILE = "modules/spam_tuner.py"

# Pipeline outputs
SERVING_MODEL_DIR = os.path.join(OUTPUT_BASE, 'serving_model')
PIPELINE_ROOT = os.path.join(OUTPUT_BASE, PIPELINE_NAME)
METADATA_PATH = os.path.join(PIPELINE_ROOT, 'metadata.sqlite')


def init_components(data_root: Text,
                    transform_module: Text,
                    training_module: Text,
                    tuner_module: Text,
                    serving_model_dir: Text):
    """Initialize the TFX components for the pipeline."""

    output = example_gen_pb2.Output(
        split_config=example_gen_pb2.SplitConfig(splits=[
            example_gen_pb2.SplitConfig.Split(name="train", hash_buckets=8),
            example_gen_pb2.SplitConfig.Split(name="eval", hash_buckets=2)
        ])
    )

    example_gen = CsvExampleGen(input_base=data_root, output_config=output)

    statistics_gen = StatisticsGen(examples=example_gen.outputs['examples'])

    schema_gen = SchemaGen(statistics=statistics_gen.outputs['statistics'])

    example_validator = ExampleValidator(
        statistics=statistics_gen.outputs['statistics'],
        schema=schema_gen.outputs['schema']
    )

    transform = Transform(
        examples=example_gen.outputs['examples'],
        schema=schema_gen.outputs['schema'],
        module_file=os.path.abspath(transform_module)
    )

    tuner = Tuner(
        module_file=os.path.abspath(tuner_module),
        examples=transform.outputs['transformed_examples'],
        transform_graph=transform.outputs['transform_graph'],
        train_args=trainer_pb2.TrainArgs(splits=['train']),
        eval_args=trainer_pb2.EvalArgs(splits=['eval']),
        tune_args=tuner_pb2.TuneArgs(num_trials=10)
    )

    trainer = Trainer(
        module_file=os.path.abspath(training_module),
        examples=transform.outputs['transformed_examples'],
        transform_graph=transform.outputs['transform_graph'],
        schema=schema_gen.outputs['schema'],
        train_args=trainer_pb2.TrainArgs(splits=['train']),
        eval_args=trainer_pb2.EvalArgs(splits=['eval']),
        hyperparameters=tuner.outputs['best_hyperparameters']
    )

    model_resolver = Resolver(
        strategy_class=LatestBlessedModelStrategy,
        model=Channel(type=Model),
        model_blessing=Channel(type=ModelBlessing)
    ).with_id('Latest_blessed_model_resolver')

    eval_config = tfma.EvalConfig(
        model_specs=[tfma.ModelSpec(label_key='Class')],
        slicing_specs=[tfma.SlicingSpec()],
        metrics_specs=[
            tfma.MetricsSpec(metrics=[
                tfma.MetricConfig(class_name='ExampleCount'),
                tfma.MetricConfig(class_name='AUC'),
                tfma.MetricConfig(class_name='FalsePositives'),
                tfma.MetricConfig(class_name='TruePositives'),
                tfma.MetricConfig(class_name='FalseNegatives'),
                tfma.MetricConfig(class_name='TrueNegatives'),
                tfma.MetricConfig(class_name='BinaryAccuracy',
                                  threshold=tfma.MetricThreshold(
                                      value_threshold=tfma.GenericValueThreshold(
                                          lower_bound={'value': 0.5}
                                      ),
                                      change_threshold=tfma.GenericChangeThreshold(
                                          direction=tfma.MetricDirection.HIGHER_IS_BETTER,
                                          absolute={'value': 0.0001}
                                      )
                                  )
                                  )
            ])
        ]
    )

    evaluator = Evaluator(
        examples=example_gen.outputs['examples'],
        model=trainer.outputs['model'],
        baseline_model=model_resolver.outputs['model'],
        eval_config=eval_config
    )

    pusher = Pusher(
        model=trainer.outputs['model'],
        model_blessing=evaluator.outputs['blessing'],
        push_destination=pusher_pb2.PushDestination(
            filesystem=pusher_pb2.PushDestination.Filesystem(
                base_directory=serving_model_dir
            )
        )
    )

    return [
        example_gen, statistics_gen, schema_gen, example_validator, transform,
        tuner, trainer, model_resolver, evaluator, pusher
    ]


def init_local_pipeline(components, pipeline_root: Text) -> pipeline.Pipeline:
    """Initialize and run a local TFX pipeline using the DirectRunner.

    Args:
        components: List of TFX components to be included in the pipeline.
        pipeline_root (Text): Root directory for storing pipeline artifacts and metadata.

    Returns:
        pipeline.Pipeline: TFX pipeline configured for local execution.
    """

    logging.info(f"Pipeline root set to: {pipeline_root}")

    return pipeline.Pipeline(
        pipeline_name=PIPELINE_NAME,
        pipeline_root=pipeline_root,
        components=components,
        enable_cache=True,
        metadata_connection_config=metadata.sqlite_metadata_connection_config(
            METADATA_PATH)
    )


if __name__ == "__main__":
    logging.set_verbosity(logging.INFO)

    components = init_components(
        DATA_ROOT,
        transform_module=TRANSFORM_MODULE_FILE,
        training_module=TRAINER_MODULE_FILE,
        tuner_module=TUNER_MODULE_FILE,
        serving_model_dir=SERVING_MODEL_DIR
    )

    pipeline = init_local_pipeline(components, PIPELINE_ROOT)
    BeamDagRunner().run(pipeline=pipeline)

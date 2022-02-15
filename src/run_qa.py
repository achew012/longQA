import hydra
import os
from model import NERLongformerQA
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from clearml import Task


def train(cfg, task) -> None:
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath="./",
        filename="best_ner_model",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        save_weights_only=True,
        period=5
    )

    model = NERLongformerQA(cfg, task)
    trainer = pl.Trainer(gpus=1, max_epochs=cfg.num_epochs,
                         callbacks=[checkpoint_callback])
    trainer.fit(model)
    return model


def test(cfg) -> None:
    model = NERLongformerQA.load_from_checkpoint(
        cfg.trained_model_path, params=cfg)

    trainer = pl.Trainer(gpus=1, max_epochs=cfg.num_epochs)
    results = trainer.test(model)
    return results


@hydra.main(config_path=os.path.join(".", "config"), config_name="config")
def hydra_main(cfg) -> float:

    print("Detected config file, initiating task... {}".format(cfg))

    if cfg.train:
        task = Task.init(project_name='LangGen', task_name='MRC-NER-train',
                         output_uri="s3://experiment-logging/storage/")
    else:
        task = Task.init(project_name='LangGen', task_name='MRC-NER-predict',
                         output_uri="s3://experiment-logging/storage/")

    task.connect(cfg)
    task.set_base_docker("nvidia/cuda:11.4.0-runtime-ubuntu20.04")
    task.execute_remotely(queue_name="compute", exit_process=True)

    if cfg.train:
        model = train(cfg, task)

    if cfg.test:
        results = test(cfg)

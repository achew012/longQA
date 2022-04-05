from common.utils import *
from transformers import AutoTokenizer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import pytorch_lightning as pl
from data.data import NERDataset
from model.model import NERLongformerQA
from torch.utils.data import DataLoader
import os
import ast
from typing import Dict, Any, List, Tuple
from omegaconf import OmegaConf
import hydra
from clearml import Task, StorageManager, Dataset as ClearML_Dataset

Task.force_requirements_env_freeze(
    force=True, requirements_file="requirements.txt")
Task.add_requirements("git+https://github.com/huggingface/datasets.git")
Task.add_requirements("hydra-core")
Task.add_requirements("pytorch-lightning")
Task.add_requirements("jsonlines")


def get_clearml_params(task: Task) -> Dict[str, Any]:
    """
    returns task params as a dictionary
    the values are casted in the required Python type
    """
    string_params = task.get_parameters_as_dict()
    clean_params = {}
    for k, v in string_params["General"].items():
        try:
            # ast.literal eval cannot read empty strings + actual strings
            # i.e. ast.literal_eval("True") -> True, ast.literal_eval("i am cute") -> error
            clean_params[k] = ast.literal_eval(v)
        except:
            # if exception is triggered, it's an actual string, or empty string
            clean_params[k] = v
    return OmegaConf.create(clean_params)


def get_dataloader(split_name, cfg) -> DataLoader:
    """Get training and validation dataloaders"""
    clearml_data_object = ClearML_Dataset.get(
        dataset_name=cfg.clearml_dataset_name,
        dataset_project=cfg.clearml_dataset_project_name,
        dataset_tags=list(cfg.clearml_dataset_tags),
        # only_published=True,
    )
    dataset_path = clearml_data_object.get_local_copy()

    # dataset_split = read_json_multiple_templates(
    #     os.path.join(dataset_path, "{}.json".format(split_name))
    # )

    dataset_split = read_json(os.path.join(
        dataset_path, "{}.json".format(split_name)))

    if cfg.debug:
        dataset_split = dataset_split[:25]

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, use_fast=True)
    dataset = NERDataset(dataset=dataset_split, tokenizer=tokenizer, cfg=cfg)

    if split_name in ["dev", "test"]:
        return DataLoader(
            dataset,
            batch_size=cfg.template_size,
            num_workers=cfg.num_workers,
            collate_fn=NERDataset.collate_fn,
            # shuffle=True,
        )
    else:
        return DataLoader(
            dataset,
            batch_size=cfg.template_size,
            num_workers=cfg.num_workers,
            collate_fn=NERDataset.collate_fn,
            # shuffle=True,
        )


def train(cfg, task) -> NERLongformerQA:
    callbacks = []

    if cfg.checkpointing:
        checkpoint_callback = ModelCheckpoint(
            dirpath="./",
            filename="best_ner_model",
            monitor="val_loss",
            mode="min",
            save_top_k=1,
            save_weights_only=True,
            every_n_epochs=cfg.every_n_epochs,
        )
        callbacks.append(checkpoint_callback)

    if cfg.early_stopping:
        early_stop_callback = EarlyStopping(
            monitor="val_loss", min_delta=0.00, patience=5, verbose=True, mode="min"
        )
        callbacks.append(early_stop_callback)

    train_loader = get_dataloader("train", cfg)
    val_loader = get_dataloader("dev", cfg)

    model = NERLongformerQA(cfg, task)
    trainer = pl.Trainer(
        gpus=cfg.gpu,
        max_epochs=cfg.num_epochs,
        accumulate_grad_batches=cfg.grad_accum,
        callbacks=callbacks,
        deterministic=True,
    )
    trainer.fit(model, train_loader, val_loader)
    return model


def test(cfg, model) -> List:
    test_loader = get_dataloader("test", cfg)
    trainer = pl.Trainer(
        gpus=cfg.gpu, max_epochs=cfg.num_epochs, deterministic=True)
    results = trainer.test(model, test_loader)
    return results


@hydra.main(config_path=os.path.join("..", "config"), config_name="config")
def hydra_main(cfg) -> float:

    pl.seed_everything(cfg.seed, workers=True)

    tags = list(cfg.task_tags) + \
        ["debug"] if cfg.debug else list(cfg.task_tags)
    tags = (
        tags + ["squad-pretrained"]
        if cfg.model_name == "mrm8488/longformer-base-4096-finetuned-squadv2"
        else tags + ["longformer-base"]
    )
    tags = tags + ["w_prompt_qns"] if cfg.add_prompt_qns else tags

    if cfg.train:
        task = Task.init(
            project_name="LongQA",
            task_name="ConversationalQA-NER-train",
            output_uri="s3://experiment-logging/storage/",
            tags=tags,
        )
    else:
        task = Task.init(
            project_name="LongQA",
            task_name="ConversationalQA-NER-predict",
            output_uri="s3://experiment-logging/storage/",
            tags=tags,
        )

    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    task.connect(cfg_dict)
    cfg = get_clearml_params(task)
    print("Detected config file, initiating task... {}".format(cfg))

    if cfg.remote:
        task.set_base_docker("nvidia/cuda:11.4.0-runtime-ubuntu20.04")
        task.execute_remotely(queue_name=cfg.queue, exit_process=True)

    if cfg.train:
        model = train(cfg, task)

    if cfg.test:
        if cfg.trained_model_path:
            trained_model_path = StorageManager.get_local_copy(
                cfg.trained_model_path)
            model = NERLongformerQA.load_from_checkpoint(
                trained_model_path, cfg=cfg, task=task
            )

        results = test(cfg, model)


if __name__ == "__main__":
    hydra_main()

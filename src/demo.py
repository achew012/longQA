#import streamlit as st
import os
import ast
from typing import Dict, Any
from omegaconf import OmegaConf
import hydra
import pandas as pd
import numpy as np
from model.model import NERLongformerQA
from data.data import NERDataset
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from clearml import StorageManager, Model, Task

import ipdb


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


question = "who are the perpetrator individuals entities?"
context = '''LONDON: A Russian delegate to talks with Ukraine was quoted on Sunday (Mar 13) as saying they had made significant progress and it was possible the delegations could soon reach draft agreements, although he did not say what these would cover. RIA news agency quoted Leonid Slutsky as comparing the state of the talks now with the situation when they first started, and saying there was "substantial progress". His comments came on day 18 of the war which began when Russian forces invaded Ukraine on Feb 24 in what the Kremlin terms a special military operation. "According to my personal expectations, this progress may grow in the coming days into a joint position of both delegations, into documents for signing," Slutsky said.
It was not clear what the scope of any such documents might be. Ukraine has said it is willing to negotiate, but not to surrender or accept any ultimatums. Three rounds of talks between the two sides in Belarus, most recently last Monday, had focused mainly on humanitarian issues and led to the limited opening of some corridors for civilians to escape fighting. Russian President Vladimir Putin said on Friday there had been some "positive shifts" in the talks, but did not elaborate. On Saturday the Kremlin said the discussions between Russian and Ukrainian officials had been continuing "in video format".'''

task = Task.get_task(task_id="911988b9d0484dcdb25534cc6ae5a964")
model_path = task.get_models()["output"][0].get_local_copy()
cfg = get_clearml_params(task)

base_model = NERLongformerQA.load_from_checkpoint(
    model_path, cfg=cfg, task=task
)

batch = [{"docid": 0, "doctext": context, "qns": question}]
batch_data = NERDataset(
    dataset=batch, tokenizer=base_model.tokenizer, cfg=cfg)

inference_dataloader = DataLoader(
    batch_data,
    batch_size=4,
    num_workers=1,
    collate_fn=NERDataset.collate_inference_fn,
)
trainer = pl.Trainer(gpus=1)
predictions = trainer.predict(base_model, dataloaders=inference_dataloader)
ipdb.set_trace()

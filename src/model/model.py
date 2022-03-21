import os
from torch.utils.data import DataLoader
from torch import nn
import torch
from typing import List, Any, Dict

# from data import NERDataset
from common.utils import *
from metric.eval import eval_ceaf
from transformers import (
    AutoTokenizer,
    AutoModelForQuestionAnswering,
    AutoConfig,
    AutoModelForSeq2SeqLM,
    set_seed,
    get_linear_schedule_with_warmup,
)
import pytorch_lightning as pl
from clearml import StorageManager, Dataset as ClearML_Dataset
import ipdb


class NERLongformerQA(pl.LightningModule):
    """Pytorch Lightning module. It wraps up the model, data loading and training code"""

    def __init__(self, cfg, task):
        """Loads the model, the tokenizer and the metric."""
        super().__init__()
        self.cfg = cfg
        self.task = task
        self.clearml_logger = self.task.get_logger()

        clearml_data_object = ClearML_Dataset.get(
            dataset_name=self.cfg.clearml_dataset_name,
            dataset_project=self.cfg.clearml_dataset_project_name,
            dataset_tags=list(self.cfg.clearml_dataset_tags),
            only_published=False,
        )
        self.dataset_path = clearml_data_object.get_local_copy()

        print("CUDA available: ", torch.cuda.is_available())

        # Load and update config then load a pretrained LEDForConditionalGeneration
        self.base_qa_model_config = AutoConfig.from_pretrained(
            self.cfg.model_name)
        # Load tokenizer and metric
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.cfg.model_name, use_fast=True
        )
        self.base_qa_model = AutoModelForQuestionAnswering.from_pretrained(
            self.cfg.model_name, config=self.base_qa_model_config
        )

        if cfg.grad_ckpt:
            self.base_qa_model.longformer.gradient_checkpointing_enable()

        self.history_embedding = torch.nn.Embedding(
            self.cfg.max_input_len, self.base_qa_model_config.hidden_size).to(self.device)

        # self.frozen_qa_model = AutoModelForQuestionAnswering.from_pretrained(
        #     "mrm8488/longformer-base-4096-finetuned-squadv2")

        # self.qg_tokenizer = AutoTokenizer.from_pretrained(
        #     "iarfmoose/t5-base-question-generator")

        # self.qg_model = AutoModelForSeq2SeqLM.from_pretrained(
        #     "iarfmoose/t5-base-question-generator")

    def _set_global_attention_mask(self, input_ids):
        """Configure the global attention pattern based on the self.task"""

        # Local attention everywhere - no global attention
        global_attention_mask = torch.zeros(
            input_ids.shape, dtype=torch.long, device=input_ids.device
        )

        # Gradient Accumulation caveat 1:
        # For gradient accumulation to work, all model parameters should contribute
        # to the computation of the loss. Remember that the self-attention layers in the LED model
        # have two sets of qkv layers, one for local attention and another for global attention.
        # If we don't use any global attention, the global qkv layers won't be used and
        # PyTorch will throw an error. This is just a PyTorch implementation limitation
        # not a conceptual one (PyTorch 1.8.1).
        # The following line puts global attention on the <s> token to make sure all model
        # parameters which is necessery for gradient accumulation to work.
        # global_attention_mask[:, :1] = 1

        # # Global attention on the first 100 tokens
        # global_attention_mask[:, :100] = 1

        # # Global attention on periods
        # global_attention_mask[(input_ids == self.tokenizer.convert_tokens_to_ids('.'))] = 1

        # Sets the questions for global attention
        batch_size = input_ids.size()[0]
        question_separators = (input_ids == 2).nonzero(as_tuple=True)
        sep_indices_batch = [
            torch.masked_select(
                question_separators[1], torch.eq(
                    question_separators[0], batch_num)
            )[0]
            for batch_num in range(batch_size)
        ]

        for batch_num in range(batch_size):
            global_attention_mask[batch_num,
                                  : sep_indices_batch[batch_num]] = 1

        return global_attention_mask

    def get_event_embeddings(self, input_ids: torch.tensor, context_mask: torch.tensor) -> torch.tensor:
        context_mask = context_mask > 0
        context = self.tokenizer.decode(
            torch.masked_select(input_ids, context_mask), skip_special_tokens=True)
        event_encodings = self.tokenizer(
            "what is the event that happened?", context, padding="max_length", truncation=True,
            max_length=self.cfg.max_input_len, return_tensors="pt")
        event_encodings = event_encodings.to(self.device)
        event_outputs = self.base_qa_model(
            input_ids=event_encodings["input_ids"], attention_mask=event_encodings["attention_mask"], output_hidden_states=True)
        event_embeddings = event_outputs.hidden_states[-1]
        return event_embeddings

    def sequence_selection_module(self, input_ids, attention_mask, event_embeddings, context_mask):
        indices_first_interval = torch.tensor([0, 1, 2]).to(self.device)
        indices_second_interval = torch.tensor([3, 4, 5]).to(self.device)
        input_ids_first_interval = torch.index_select(
            input_ids, 0, indices_first_interval)
        attention_mask_first_interval = torch.index_select(
            attention_mask, 0, indices_first_interval)

        input_ids_second_interval = torch.index_select(
            input_ids, 0, indices_second_interval)
        attention_mask_second_interval = torch.index_select(
            attention_mask, 0, indices_second_interval)

        single_layer = self.base_qa_model(
            input_ids=input_ids_first_interval,
            attention_mask=attention_mask_first_interval,  # mask padding tokens
            output_hidden_states=True,
        )
        start_logits = single_layer.start_logits
        end_logits = single_layer.end_logits
        #logits = torch.nn.functional.softmax(single_layer.hidden_states[-1])

        candidate_start_tokens = torch.topk(start_logits, 5).indices
        candidate_end_tokens = torch.topk(start_logits, 5).indices

        span_embeds = torch.matmul(self.history_embedding(
            candidate_start_tokens), torch.transpose(self.history_embedding(candidate_end_tokens), 1, 2))

        ipdb.set_trace()

        # redefine the batches
        # Take different sequence length and order and add the embeddings
        return None

    def forward(self, **batch):
        input_ids, attention_mask = (
            batch["input_ids"],
            batch["attention_mask"],
        )

        # what is the event that happened?
        event_embeddings = self.get_event_embeddings(
            input_ids, batch["context_mask"])

        # input_embeds
        qa_embeddings = self.base_qa_model.longformer.embeddings(input_ids)

        # combine input_embeds with event embeds
        combined_embeddings = torch.add(qa_embeddings, event_embeddings)

        # Add a new init history embedding here
        # Randomly generate different sequences of conversations
        # combined_embeddings = self.sequence_selection_module(
        #     input_ids, attention_mask, event_embeddings, batch["context_mask"])

        if "start_positions" in batch.keys():

            start, end = batch["start_positions"], batch["end_positions"]

            outputs = self.base_qa_model(
                # input_ids=input_ids,
                inputs_embeds=combined_embeddings,
                attention_mask=attention_mask,  # mask padding tokens
                global_attention_mask=self._set_global_attention_mask(
                    input_ids),
                start_positions=start,
                end_positions=end,
                output_hidden_states=True,
            )

        else:
            outputs = self.base_qa_model(
                # input_ids=input_ids,
                inputs_embeds=combined_embeddings,
                attention_mask=attention_mask,  # mask padding tokens
            )

        return outputs

    def training_step(self, batch, batch_nb):
        """Call the forward pass then return loss"""
        docids = batch.pop("docid", None)
        gold_mentions = batch.pop("gold_mentions", None)

        outputs = self.forward(**batch)
        return {"loss": outputs.loss}

    def training_epoch_end(self, outputs):
        total_loss = []
        for batch in outputs:
            total_loss.append(batch["loss"])
        self.log("train_loss", sum(total_loss) / len(total_loss))

    def extract_answer_spans(self,
                             start_candidates,
                             start_candidates_logits,
                             end_candidates,
                             end_candidates_logits,
                             question_indices,
                             tokens,
                             attention_mask
                             ) -> List:

        valid_candidates = []
        # For each candidate in sample
        for start_index, start_score in zip(
            start_candidates, start_candidates_logits
        ):
            for end_index, end_score in zip(end_candidates, end_candidates_logits):

                # throw out invalid predictions
                if start_index in question_indices:
                    continue
                elif end_index in question_indices:
                    continue
                elif end_index < start_index:
                    continue
                elif (end_index - start_index) > self.cfg.max_prediction_span:
                    continue
                elif (start_index) > (torch.count_nonzero(attention_mask)):
                    continue

                if start_index == 0:
                    if len(valid_candidates) < 1:
                        valid_candidates.append(
                            (start_index.item(), end_index.item(), "", 0)
                        )
                else:
                    valid_candidates.append(
                        (
                            start_index.item(),
                            end_index.item(),
                            self.tokenizer.decode(
                                tokens[start_index:end_index], skip_special_tokens=True),
                            (start_score + end_score).item(),
                        )
                    )
        return valid_candidates

    def _evaluation_step(self, split, batch, batch_nb):
        """Validaton or Testing - predict output, compare it with gold, compute rouge1, 2, L, and log result"""
        # input_ids, attention_mask, start, end  = batch["input_ids"], batch["attention_mask"], batch["start_positions"], batch["end_positions"]

        docids = batch.pop("docid", None)
        gold_mentions = batch.pop("gold_mentions", None)

        outputs = self(**batch)  # mask padding tokens

        candidates_start_batch = torch.topk(outputs.start_logits, 20)
        candidates_end_batch = torch.topk(outputs.end_logits, 20)

        # Get qns mask
        batch_size = batch["input_ids"].size()[0]
        question_separators = (batch["input_ids"] == 2).nonzero(as_tuple=True)
        sep_indices_batch = [
            torch.masked_select(
                question_separators[1], torch.eq(
                    question_separators[0], batch_num)
            )[0]
            for batch_num in range(batch_size)
        ]
        question_indices_batch = [
            [i + 1 for i, token in enumerate(tokens[1: sep_idx + 1])]
            for tokens, sep_idx in zip(batch["input_ids"], sep_indices_batch)
        ]
        batch_outputs = []

        # For each sample in batch
        for (
            start_candidates,
            start_candidates_logits,
            end_candidates,
            end_candidates_logits,
            tokens,
            question_indices,
            start_gold,
            end_gold,
            attention_mask,
            docid,
            gold_mention,
        ) in zip(
            candidates_start_batch.indices,
            candidates_start_batch.values,
            candidates_end_batch.indices,
            candidates_end_batch.values,
            batch["input_ids"],
            question_indices_batch,
            batch["start_positions"],
            batch["end_positions"],
            batch["attention_mask"],
            docids,
            gold_mentions,
        ):
            valid_candidates = self.extract_answer_spans(start_candidates,
                                                         start_candidates_logits,
                                                         end_candidates,
                                                         end_candidates_logits,
                                                         question_indices,
                                                         tokens,
                                                         attention_mask
                                                         )

            # Attempt to implement a confidence threshold
            score_list = [candidate[3] for candidate in valid_candidates]

            # top_5 = sorted(score_list, reverse=True)[:5]
            threshold = max(score_list) if len(score_list) > 0 else 0

            batch_outputs.append(
                {
                    "docid": docid,
                    "qns": self.tokenizer.decode(tokens[1: len(question_indices)]),
                    "gold_mention": gold_mention,
                    "context": self.tokenizer.decode(
                        torch.masked_select(tokens, torch.gt(attention_mask, 0))[
                            1 + len(question_indices):
                        ],
                        skip_special_tokens=True
                    ),
                    "start_gold": start_gold.item(),
                    "end_gold": end_gold.item(),
                    "gold": self.tokenizer.decode(tokens[start_gold:end_gold], skip_special_tokens=True),
                    "candidates": [
                        candidate
                        for candidate in valid_candidates
                        if candidate[3] == threshold
                    ],
                }
            )

        logs = {"loss": outputs.loss}

        return {"loss": logs["loss"], "preds": batch_outputs}

    def validation_step(self, batch, batch_nb):
        out = self._evaluation_step("val", batch, batch_nb)
        return {"results": out["preds"], "loss": out["loss"]}

    def validation_epoch_end(self, outputs):
        total_loss = []
        for batch in outputs:
            total_loss.append(batch["loss"])
        self.log("val_loss", sum(total_loss) / len(total_loss))

    def test_step(self, batch, batch_nb):
        out = self._evaluation_step("test", batch, batch_nb)
        return {"results": out["preds"]}

    #################################################################################
    def test_epoch_end(self, outputs):
        logs = {}
        doctexts_tokens, golds = read_golds_from_test_file(
            self.dataset_path, self.tokenizer, self.cfg
        )

        # Consolidate all the batches to single variable
        predictions = {}
        for batch in outputs:
            for sample in batch["results"]:

                if sample["docid"] not in predictions.keys():
                    predictions[sample["docid"]] = {
                        "docid": sample["docid"],
                        "context": sample["context"],
                        "qns": [sample["qns"]],
                        "gold_mention": [sample["gold_mention"]],
                        "gold": [sample["gold"]],
                        "candidates": [sample["candidates"]],
                    }
                else:
                    predictions[sample["docid"]]["qns"].append(sample["qns"])
                    predictions[sample["docid"]]["gold_mention"].append(
                        sample["gold_mention"]
                    )
                    predictions[sample["docid"]]["gold"].append(sample["gold"])
                    predictions[sample["docid"]]["candidates"].append(
                        sample["candidates"]
                    )

        # Convert to evaluation format
        preds = OrderedDict()
        for key, doc in predictions.items():
            if key not in preds:
                preds[key] = OrderedDict()
                for idx, role in enumerate(self.cfg.role_map.keys()):
                    preds[key][role] = []
                    if idx + 1 > len(doc["candidates"]):
                        continue
                    elif doc["candidates"][idx]:

                        # if doc["candidates"][idx][0][2]!="</s>":
                        #     filtered_candidates = set(ent_spans).intersection(set(candidate[2].replace("</s>", "").strip() for candidate in doc["candidates"][idx]))
                        #     filtered_candidates = filtered_candidates.union(set(candidate[2].replace("</s>", "").strip() for candidate in doc["candidates"][idx]))
                        # else:
                        #     filtered_candidates = []
                        # preds[key][role] = [[candidate] for candidate in filtered_candidates]

                        filtered_candidates = doc["candidates"][idx]
                        for candidate in filtered_candidates:
                            if candidate[2] != '':
                                preds[key][role] = [[candidate[2]]]
            else:
                print("duplicated example")

        if self.cfg.debug:
            filtered_golds = OrderedDict()
            eval_keys = [docid for docid in preds.keys()]
            for docid in eval_keys:
                filtered_golds[docid] = golds[docid]
            golds = filtered_golds

        results = eval_ceaf(preds, golds)
        print("================= CEAF score =================")
        print(
            "phi_strict: P: {:.2f}%,  R: {:.2f}%, F1: {:.2f}%".format(
                results["strict"]["micro_avg"]["p"] * 100,
                results["strict"]["micro_avg"]["r"] * 100,
                results["strict"]["micro_avg"]["f1"] * 100,
            )
        )
        print(
            "phi_prop: P: {:.2f}%,  R: {:.2f}%, F1: {:.2f}%".format(
                results["prop"]["micro_avg"]["p"] * 100,
                results["prop"]["micro_avg"]["r"] * 100,
                results["prop"]["micro_avg"]["f1"] * 100,
            )
        )
        print("==============================================")
        logs["test_micro_avg_f1_phi_strict"] = results["strict"]["micro_avg"]["f1"]
        logs["test_micro_avg_precision_phi_strict"] = results["strict"]["micro_avg"][
            "p"
        ]
        logs["test_micro_avg_recall_phi_strict"] = results["strict"]["micro_avg"]["r"]

        self.clearml_logger.report_scalar(
            title="f1",
            series="test",
            value=logs["test_micro_avg_f1_phi_strict"],
            iteration=1,
        )
        self.clearml_logger.report_scalar(
            title="precision",
            series="test",
            value=logs["test_micro_avg_precision_phi_strict"],
            iteration=1,
        )
        self.clearml_logger.report_scalar(
            title="recall",
            series="test",
            value=logs["test_micro_avg_recall_phi_strict"],
            iteration=1,
        )

        preds_list = [{**doc, "docid": key} for key, doc in preds.items()]
        to_jsonl("./predictions.jsonl", preds_list)
        self.task.upload_artifact(
            name="predictions", artifact_object="./predictions.jsonl"
        )
        return {"results": results}

    def predict_step(self, batch, batch_nb):
        docids = batch.pop("docid", None)
        outputs = self(**batch)
        candidates_start_batch = torch.topk(outputs.start_logits, 20)
        candidates_end_batch = torch.topk(outputs.end_logits, 20)
        batch_size = batch["input_ids"].size()[0]
        question_separators = (batch["input_ids"] == 2).nonzero(as_tuple=True)
        sep_indices_batch = [
            torch.masked_select(
                question_separators[1], torch.eq(
                    question_separators[0], batch_num)
            )[0]
            for batch_num in range(batch_size)
        ]
        question_indices_batch = [
            [i + 1 for i, token in enumerate(tokens[1: sep_idx + 1])]
            for tokens, sep_idx in zip(batch["input_ids"], sep_indices_batch)
        ]
        batch_outputs = []

        # For each sample in batch
        for (
            start_candidates,
            start_candidates_logits,
            end_candidates,
            end_candidates_logits,
            tokens,
            question_indices,
            attention_mask,
            docid,
        ) in zip(
            candidates_start_batch.indices,
            candidates_start_batch.values,
            candidates_end_batch.indices,
            candidates_end_batch.values,
            batch["input_ids"],
            question_indices_batch,
            batch["attention_mask"],
            docids,
        ):
            valid_candidates = self.extract_answer_spans(start_candidates,
                                                         start_candidates_logits,
                                                         end_candidates,
                                                         end_candidates_logits,
                                                         question_indices,
                                                         tokens,
                                                         attention_mask
                                                         )

            score_list = [candidate[3] for candidate in valid_candidates]
            threshold = max(score_list)

            batch_outputs.append(
                {
                    "docid": docid,
                    "qns": self.tokenizer.decode(tokens[1: len(question_indices)]),
                    "context": self.tokenizer.decode(
                        torch.masked_select(tokens, torch.gt(attention_mask, 0))[
                            1 + len(question_indices):
                        ]
                    ),
                    "candidates": [
                        candidate
                        for candidate in valid_candidates
                        if candidate[3] == threshold
                    ],
                }
            )

        return {"results": batch_outputs}

    def configure_optimizers(self):
        """Configure the optimizer and the learning rate scheduler"""

        # # Freeze the model
        # for idx, (name, parameters) in enumerate(self.base_qa_model.named_parameters()):
        #     if idx % 2 == 0:
        #         parameters.requires_grad = False
        #     else:
        #         parameters.requires_grad = True

        optimizer = torch.optim.AdamW(self.parameters(), lr=self.cfg.lr)
        return [optimizer]

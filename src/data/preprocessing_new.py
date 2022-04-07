from typing import List, Dict, Any, Tuple
import ipdb
import random
from omegaconf import OmegaConf
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForQuestionAnswering,
)
from tqdm import tqdm


def format_qa_input(generated_question: str, context: list, qa_tokenizer):
    return qa_tokenizer(
        [generated_question] * len(context),
        context,
        truncation=True,
        padding="max_length",
        max_length=1024,
        return_tensors="pt",
    )


def process_qa(batch, qa_model, qa_tokenizer):
    outputs = qa_model(
        input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
    )
    start_scores, end_scores = torch.argmax(outputs.start_logits, dim=-1), torch.argmax(
        outputs.end_logits, dim=-1
    )

    answer_list = []
    for idx, doc_tokens in enumerate(batch["input_ids"]):
        answer_ids = doc_tokens[start_scores[idx] : end_scores[idx] + 1]
        answer = qa_tokenizer.decode(answer_ids[:25], skip_special_tokens=True)
        answer_list.append(answer.split("?")[-1])
    return answer_list


def ask(generated_question: str, context: list, qa_model, qa_tokenizer):
    batch_qa = format_qa_input(generated_question, context, qa_tokenizer)
    answer = process_qa(batch_qa, qa_model, qa_tokenizer)
    return answer


def get_question(role: str, event_mention: str = None) -> str:
    if "deaths" in role:
        role = "kia"

    elif "injured" in role:
        role = "wia"

    if event_mention:
        role = f"{role[:-1]} in {event_mention}?"

    return role


def is_existing_question(natural_question: str, qns_ans: List) -> Tuple[bool, Any]:
    for idx, question_group in enumerate(qns_ans):
        if natural_question in question_group[0]:
            return (True, idx)
    return (False, -1)


def random_swap(input_qns: str, tokenizer: Any, chance: float = 0.0):
    if random.random() < chance:
        input_ids = tokenizer.encode(input_qns)
        random_position = random.randint(0, len(input_ids) - 1)
        random_ids = random.randint(0, len(tokenizer.vocab) - 1)
        input_ids[random_position] = random_ids
        return tokenizer.decode(input_ids, skip_special_tokens=True)
    else:
        return input_qns


def generate_questions_from_template(
    doc: Dict, role_map: Dict, tokenizer: object, event_mention: str = None
) -> List[Dict]:

    events = []
    for template in doc["templates"]:
        incident = template.pop("incident_type", None)
        qns_ans = []
        for key in role_map.keys():
            natural_question = get_question(role_map[key].lower(), event_mention)
            # natural_question = random_swap(natural_question, tokenizer)
            if natural_question:
                if len(template[key]) > 0:
                    mention = template[key][0][0][0]
                    start_idx = template[key][0][0][1]
                    end_idx = start_idx + len(mention)
                else:
                    mention = ""
                    start_idx = 0
                    end_idx = 0

                has_existing_idx, existing_idx = is_existing_question(
                    natural_question, qns_ans
                )

                if start_idx == 0 and end_idx == 0:
                    # if it's a blank answer, 20% chance of being included into the training set
                    continue

                # Appends question-answer pair to list. if question exist, append mentions to it.
                if has_existing_idx:
                    if (start_idx, end_idx, mention) not in qns_ans[existing_idx][1]:
                        qns_ans[existing_idx] = [
                            qns_ans[existing_idx][0],
                            qns_ans[existing_idx][1] + [(start_idx, end_idx, mention)],
                        ]
                else:
                    qns_ans.append([natural_question, [(start_idx, end_idx, mention)]])

        events.append({"incident": incident, "question_answer_pair_list": qns_ans})

    return events


def convert_char_indices(
    ans_char_start: int,
    ans_char_end: int,
    spans_list: List[List[Tuple[int, int]]],
    max_idx: int,
) -> List:
    # offset has to be List[List[int, int]] or tensor of same shape
    # if char indices more than end idx in last word span, reset indices to 0

    if ans_char_end > max_idx or ans_char_start > max_idx:
        ans_char_start = 0
        ans_char_end = 0

    if ans_char_start == 0 and ans_char_end == 0:
        token_span = [0, 0]
    else:
        token_span = []
        for idx, span in enumerate(spans_list):
            if (
                ans_char_start >= span[0]
                and ans_char_start <= span[1]
                and len(token_span) == 0
            ):
                token_span.append(idx)

            if (
                ans_char_end >= span[0]
                and ans_char_end <= span[1]
                and len(token_span) == 1
            ):
                token_span.append(idx)
                break

        # if token span is incomplete
        if len(token_span) != 2:
            print("cant find token span")
            ipdb.set_trace()

    return token_span


def convert_character_spans_to_word_spans(
    processed_dataset: List, doc: Dict, events: List[Dict], tokenizer: Any, cfg: Any
) -> Dict:

    docid = doc["docid"]
    context = doc["doctext"]

    for event in events[:1]:
        incident = event["incident"]
        question_answer_pair_list = event["question_answer_pair_list"]

        for question_answer_pair in question_answer_pair_list:
            processed_sample = {}
            qns = question_answer_pair[0]
            answers = question_answer_pair[1]

            if len(answers) == 0:
                continue

            context_encodings = tokenizer(
                qns,
                context,
                padding="max_length",
                truncation=True,
                max_length=cfg.max_input_len,
                return_offsets_mapping=True,
                return_tensors="pt",
            )
            sequence_ids = context_encodings.sequence_ids()

            # Detect if the answer is out of the span (in which case this feature is labeled with the CLS index).
            qns_offset = sequence_ids.index(1) - 1
            pad_start_idx = sequence_ids[sequence_ids.index(1) :].index(None)
            offsets_wo_pad = context_encodings["offset_mapping"][0][
                qns_offset:pad_start_idx
            ]
            context_mask = context_encodings["attention_mask"]
            context_mask[0][: qns_offset + 1] = 0

            processed_sample["docid"] = docid
            processed_sample["context"] = context
            processed_sample["input_ids"] = context_encodings["input_ids"].squeeze(0)

            processed_sample["attention_mask"] = context_encodings[
                "attention_mask"
            ].squeeze(0)
            processed_sample["context_mask"] = context_mask.squeeze(0)
            processed_sample["qns"] = qns

            mentions = []
            start_spans = []
            end_spans = []

            for ans_char_start, ans_char_end, mention in answers:

                token_span = convert_char_indices(
                    ans_char_start,
                    ans_char_end,
                    offsets_wo_pad,
                    max_idx=offsets_wo_pad[-1][1],
                )

                mentions.append(mention)
                start_spans.append(token_span[0] + qns_offset)
                end_spans.append(token_span[1] + qns_offset + 1)

            # Limit to only one answer for aggragated questions
            processed_sample["gold_mentions"] = mentions[0]
            processed_sample["start"] = start_spans[0]
            processed_sample["end"] = end_spans[0]

            processed_dataset.append(processed_sample)

    return processed_dataset


def get_prompt_qns(dataset: List[Dict], cfg: OmegaConf) -> List[str]:
    qa_tokenizer = AutoTokenizer.from_pretrained(
        "mrm8488/longformer-base-4096-finetuned-squadv2"
    )

    qa_model = AutoModelForQuestionAnswering.from_pretrained(
        "mrm8488/longformer-base-4096-finetuned-squadv2"
    )

    def chunks(lst: list, batch_size: int):
        """Yield successive n-sized chunks from lst."""
        for i in range(0, len(lst), batch_size):
            yield lst[i : i + batch_size]

    question_list = []
    batches = chunks(dataset, cfg.batch_size)
    for batch in tqdm(batches):
        context_batch = [doc["doctext"] for doc in batch]
        question_list += ask(
            "what is the trigger?", context_batch, qa_model, qa_tokenizer
        )

    return question_list


def process_train_data(dataset: List[Dict], tokenizer: Any, cfg: Any) -> List:
    role_map = cfg.role_map

    processed_dataset = []

    print(
        f"Preprocessing Dataset {'with Additional Prompt Questions' if cfg.add_prompt_qns else '...'}"
    )

    dataset = [doc for doc in dataset if len(doc["templates"]) > 0]

    if cfg.add_prompt_qns:
        event_list = get_prompt_qns(dataset, cfg)

        for doc, event_mention in zip(dataset, event_list):
            events = generate_questions_from_template(
                doc, role_map, tokenizer, event_mention
            )
            processed_dataset = convert_character_spans_to_word_spans(
                processed_dataset, doc, events, tokenizer, cfg
            )
    else:
        for doc in dataset:
            events = generate_questions_from_template(doc, role_map, tokenizer)
            processed_dataset = convert_character_spans_to_word_spans(
                processed_dataset, doc, events, tokenizer, cfg
            )

    return processed_dataset


def process_inference_data(dataset: List[Dict], tokenizer: Any, cfg: Any) -> Dict:

    role_map = cfg.role_map

    processed_dataset = []

    for doc in dataset:
        docid = doc["docid"]
        context = doc["doctext"]
        qns = doc["qns"]
        context_encodings = tokenizer(
            qns,
            context,
            padding="max_length",
            truncation=True,
            max_length=cfg.max_input_len,
            return_offsets_mapping=True,
            return_tensors="pt",
        )

        processed_sample = {}
        sequence_ids = context_encodings.sequence_ids()
        qns_offset = sequence_ids.index(1) - 1
        # pad_start_idx = sequence_ids[sequence_ids.index(
        #     1):].index(None)
        # offsets_wo_pad = context_encodings["offset_mapping"][0][qns_offset:pad_start_idx]
        context_mask = context_encodings["attention_mask"]
        context_mask[0][:qns_offset] = 0

        processed_sample["docid"] = docid
        processed_sample["context"] = context
        processed_sample["qns"] = docid
        processed_sample["input_ids"] = context_encodings["input_ids"].squeeze(0)
        processed_sample["attention_mask"] = context_encodings[
            "attention_mask"
        ].squeeze(0)
        processed_sample["context_mask"] = context_mask.squeeze(0)
        processed_dataset.append(processed_sample)

    return processed_dataset

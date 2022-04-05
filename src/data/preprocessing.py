from typing import List, Dict, Any, Tuple
import ipdb


def get_question(role):

    if "deaths" in role:
        role = "kia"
        return role

    elif "injured" in role:
        role = "wia"
        return role

    return role


def is_existing_question(natural_question: str, qns_ans: List) -> Tuple[bool, Any]:
    for idx, question_group in enumerate(qns_ans):
        if natural_question in question_group[0]:
            return (True, idx)
    return (False, -1)


def generate_questions_from_template(doc: Dict, role_map: Dict) -> List[Dict]:
    # Only take the 1st label of each role
    events = []
    for template in doc["templates"]:
        print('docid: ', doc["docid"])
        incident = template.pop('incident_type', None)
        qns_ans = []
        for key in role_map.keys():
            natural_question = get_question(role_map[key].lower())
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
                    natural_question, qns_ans)

                if start_idx == 0 and end_idx == 0:
                    # if it's a blank answer, 20% chance of being included into the training set
                    continue

                # Appends question-answer pair to list. if question exist, append mentions to it.
                if has_existing_idx:
                    if (start_idx, end_idx, mention) not in qns_ans[existing_idx][1]:
                        qns_ans[existing_idx] = [qns_ans[existing_idx][0], qns_ans[existing_idx][1] + [
                            (start_idx, end_idx, mention)]]
                else:
                    qns_ans.append(
                        [natural_question, [(start_idx, end_idx, mention)]])
        print('qns_ans: ', qns_ans)
        events.append(
            {"incident": incident, "question_answer_pair_list": qns_ans})

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
                span[0] <= ans_char_start <= span[1]
                    and len(token_span) == 0
            ):
                token_span.append(idx)

            if (
                span[0] <= ans_char_end <= span[1]
                    and len(token_span) == 1
            ):
                token_span.append(idx)
                break

        # if token span is incomplete
        if len(token_span) != 2:
            print("cant find token span")
            ipdb.set_trace()

    return token_span


def convert_character_spans_to_word_spans(processed_dataset: Dict, doc: Dict, events: List[Dict], tokenizer: Any, cfg: Any) -> Dict:

    docid = doc["docid"]
    context = doc["doctext"]

    for event in events:
        incident = event["incident"]
        question_answer_pair_list = event["question_answer_pair_list"]

        for question_answer_pair in question_answer_pair_list:
            qns = question_answer_pair[0]
            answers = question_answer_pair[1]

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
            # context_mask = context_encodings["attention_mask"]
            # context_mask[0][: qns_offset + 1] = 0

            processed_dataset["docid"].append(docid)
            processed_dataset["context"].append(context)
            processed_dataset["input_ids"].append(
                context_encodings["input_ids"].squeeze(0)
            )
            processed_dataset["attention_mask"].append(
                context_encodings["attention_mask"].squeeze(0)
            )
            # processed_dataset["context_mask"].append(context_mask.squeeze(0))
            processed_dataset["qns"].append(qns)

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
            processed_dataset["gold_mentions"].append(mentions[0])
            processed_dataset["start"].append(start_spans[0])
            processed_dataset["end"].append(end_spans[0])

    return processed_dataset


def process_train_data(dataset: List[Dict], tokenizer: Any, cfg: Any) -> Dict:
    role_map = cfg.role_map

    processed_dataset = {
        "docid": [],
        "context": [],
        "input_ids": [],
        "attention_mask": [],
        "qns": [],
        "gold_mentions": [],
        "start": [],
        "end": []
    }

    for doc in dataset:
        qns_ans = generate_questions_from_template(doc, role_map)
        processed_dataset = convert_character_spans_to_word_spans(
            processed_dataset, doc, qns_ans, tokenizer, cfg)

    return processed_dataset


def process_inference_data(dataset: List[Dict], tokenizer: Any, cfg: Any) -> Dict:
    role_map = cfg.role_map

    processed_dataset = {
        "docid": [],
        "context": [],
        "qns": [],
        "input_ids": [],
        "attention_mask": [],
        # "gold_mentions": [],
        # "start": [],
        # "end": []
    }

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
            return_tensors="pt"
        )

        processed_dataset["docid"].append(docid)
        processed_dataset["context"].append(context)
        processed_dataset["qns"].append(docid)
        processed_dataset["input_ids"].append(
            context_encodings["input_ids"].squeeze(0))
        processed_dataset["attention_mask"].append(
            context_encodings["attention_mask"].squeeze(0))

    return processed_dataset

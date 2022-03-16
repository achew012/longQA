from typing import List, Dict, Any, Tuple
from xmlrpc.client import Boolean
import ipdb


def get_question(role):
    question_dict = {
        "event": "what is the event that happened?",
        "target": "what was the target of the event?",
        "victim": "who is the victim?",
        "perpetrator individuals": "who attacked the victim?",
        "weapons": "what weapon was used to attack the victim?",
        "kia": "how many victims were killed?",
        "wia": "how many victims were injured?",
        "perpetrator organizations": "what organization did the attacker belong to?",
        "location": "where did the event happen?",
        "civilian targets": "Were there civilian victims?",
        "government official targets": "Were government officials attacked?",
        "military targets": "Were there military casualties?",
    }

    if "deaths" in role:
        role = "kia"
        return question_dict[role]

    elif "injured" in role:
        role = "wia"
        return question_dict[role]

    elif role in question_dict.keys():
        return question_dict[role]

    else:
        return None


def is_existing_question(natural_question: str, qns_ans: List) -> Tuple[bool, Any]:
    for idx, question_group in enumerate(qns_ans):
        if natural_question in question_group[0]:
            return (True, idx)
    return (False, -1)


def generate_questions_from_template(doc: Dict, role_map: Dict) -> List[List]:
    qns_ans = []
    for key in doc["templates"].keys():

        # ipdb.set_trace()

        natural_question = get_question(role_map[key].lower())
        if natural_question:
            if len(doc["templates"][key]) > 0:
                mention = doc["templates"][key][0][0][0]
                start_idx = doc["templates"][key][0][0][1]
                end_idx = start_idx + len(mention)
            else:
                mention = ""
                start_idx = 0
                end_idx = 0

            has_existing_idx, existing_idx = is_existing_question(
                natural_question, qns_ans)
            # Appends question-answer pair to list. if question exist, append mentions to it.
            if has_existing_idx:
                if (start_idx, end_idx, mention) not in qns_ans[existing_idx][1]:
                    qns_ans[existing_idx] = [qns_ans[existing_idx][0], qns_ans[existing_idx][1] + [
                        (start_idx, end_idx, mention)]]
            else:
                qns_ans.append(
                    [natural_question, [(start_idx, end_idx, mention)]])

    return qns_ans


def convert_character_spans_to_word_spans(processed_dataset: Dict, doc: Dict, qns_ans: List[List], tokenizer: Any, cfg: Any) -> Dict:
    docid = doc["docid"]
    context = doc["doctext"]

    for qns, ans_char_start, ans_char_end, mention in qns_ans:
        context_encodings = tokenizer(qns, context, padding="max_length", truncation=True,
                                      max_length=cfg.max_input_len, return_offsets_mapping=True, return_tensors="pt")
        sequence_ids = context_encodings.sequence_ids()

        # Detect if the answer is out of the span (in which case this feature is labeled with the CLS index).
        qns_offset = sequence_ids.index(1)-1
        pad_start_idx = sequence_ids[sequence_ids.index(
            1):].index(None)
        offsets_wo_pad = context_encodings["offset_mapping"][0][qns_offset:pad_start_idx]

        # if char indices more than end idx in last word span, reset indices to 0
        if ans_char_end > offsets_wo_pad[-1][1] or ans_char_start > offsets_wo_pad[-1][1]:
            ans_char_start = 0
            ans_char_end = 0

        if ans_char_start == 0 and ans_char_end == 0:
            token_span = [0, 0]
        else:
            token_span = []
            for idx, span in enumerate(offsets_wo_pad):
                if ans_char_start >= span[0] and ans_char_start <= span[1] and len(token_span) == 0:
                    token_span.append(idx)

                if ans_char_end >= span[0] and ans_char_end <= span[1] and len(token_span) == 1:
                    token_span.append(idx)
                    break

        # If token span is incomplete
        if len(token_span) < 2:
            ipdb.set_trace()
            # print("span: ", tokenizer.decode(context_encodings["input_ids"][0][token_span[0]+qns_offset:token_span[1]+1+qns_offset]))

       # BIO SCHEME - Not needed for QA
        # if token_span!=[0,0]:
        #     labels[token_span[0]:token_span[0]+1] = [self.labels2idx["B-"+qns]]
        #     if len(labels[token_span[0]+1:token_span[1]+1])>0 and (token_span[1]-token_span[0])>0:
        #         labels[token_span[0]+1:token_span[1]+1] = [self.labels2idx["I-"+qns]]*(token_span[1]-token_span[0])

        processed_dataset["docid"].append(docid)
        processed_dataset["context"].append(context)
        processed_dataset["input_ids"].append(
            context_encodings["input_ids"].squeeze(0))
        processed_dataset["attention_mask"].append(
            context_encodings["attention_mask"].squeeze(0))
        processed_dataset["qns"].append(docid)
        processed_dataset["gold_mentions"].append(mention)
        processed_dataset["start"].append(
            token_span[0]+qns_offset)
        processed_dataset["end"].append(
            token_span[1]+qns_offset+1)

    return processed_dataset


def process_train_data(dataset: List[Dict], tokenizer: Any, cfg: Any, role_map: Dict) -> Dict:
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


def process_inference_data(dataset: List[Dict], tokenizer: Any, cfg: Any, role_map: Dict) -> Dict:
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
        context_encodings = tokenizer(qns, context, padding="max_length", truncation=True,
                                      max_length=cfg.max_input_len, return_offsets_mapping=True, return_tensors="pt")

        processed_dataset["docid"].append(docid)
        processed_dataset["context"].append(context)
        processed_dataset["qns"].append(docid)
        processed_dataset["input_ids"].append(
            context_encodings["input_ids"].squeeze(0))
        processed_dataset["attention_mask"].append(
            context_encodings["attention_mask"].squeeze(0))

    return processed_dataset

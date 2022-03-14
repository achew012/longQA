from typing import List, Dict, Any
import ipdb


def generate_questions_from_template(doc: Dict, role_map: Dict) -> List[List]:
    # Only take the 1st label of each role
    # qns_ans = [["who are the {} entities?".format(role_map[key].lower()), doc["extracts"][key][0][0][1] if len(doc["extracts"][key]) > 0 else 0, doc["extracts"][key][0][0][1]+len(
    #     doc["extracts"][key][0][0][0]) if len(doc["extracts"][key]) > 0 else 0, doc["extracts"][key][0][0][0] if len(doc["extracts"][key]) > 0 else ""] for key in doc["extracts"].keys()]

    # qns_ans = [["who are the {} entities?".format(role_map[key].lower()), doc["templates"][0][key][0][0][1] if len(doc["templates"][0][key]) > 0 else 0, doc["templates"][0][key][0][0][1]+len(
    #     doc["templates"][0][key][0][0][0]) if len(doc["templates"][0][key]) > 0 else 0, doc["templates"][0][key][0][0][0] if len(doc["templates"][0][key]) > 0 else ""] for key in doc["templates"][0].keys()]

    qns_ans = [["who are the {} entities?".format(role_map[key].lower()), doc["templates"][key][0][0][1] if len(doc["templates"][key]) > 0 else 0, doc["templates"][key][0][0][1]+len(
        doc["templates"][key][0][0][0]) if len(doc["templates"][key]) > 0 else 0, doc["templates"][key][0][0][0] if len(doc["templates"][key]) > 0 else ""] for key in doc["templates"].keys()]

    # templates = doc["templates"]
    # qns_ans = []
    # for template in templates:
    #     for key in template.keys():
    #         if key != "incident_type":
    #             role = key
    #             answer = template[key][0][0][0] if len(
    #                 template[key]) > 0 else ""
    #             start_idx = template[key][0][0][1] if len(
    #                 template[key]) > 0 else 0
    #             end_idx = start_idx + len(answer)
    #             qns_ans.append(["who are the {} entities?".format(
    #                 role), start_idx, end_idx, answer])

    # expand on all labels in each role
    # qns_ans = [["who are the {} entities?".format(role_map[key].lower()), mention[1] if len(mention)>0 else 0, mention[1]+len(mention[0]) if len(mention)>0 else 0, mention[0] if len(mention)>0 else ""] for key in doc["extracts"].keys() for cluster in doc["extracts"][key] for mention in cluster]
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

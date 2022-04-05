import jsonlines
from collections import OrderedDict
import re
import json
import os
import ipdb
from typing import List, Dict, Any, Tuple


def to_jsonl(filename: str, file_obj):
    resultfile = open(filename, "wb")
    writer = jsonlines.Writer(resultfile)
    writer.write_all(file_obj)


def read_json(jsonfile):
    with open(jsonfile, "rb") as file:
        file_object = [json.loads(sample) for sample in file]
    return file_object


def write_json(filename, file_object):
    with open(filename, "w") as file:
        file.write(json.dumps(file_object))


def read_json_multiple_templates(jsonfile):
    with open(jsonfile, "rb") as file:
        file_object = [json.loads(sample) for sample in file]

        for i in range(len(file_object)):
            if len(file_object[i]["templates"]) > 0:
                file_object[i]["templates"] = file_object[i]["templates"][0]
                del file_object[i]["templates"]["incident_type"]
            else:
                file_object[i]["templates"] = {
                    "Location": [],
                    "PerpInd": [],
                    "PerpOrg": [],
                    "PhysicalTarget": [],
                    "Weapon": [],
                    "HumTargetCivilian": [],
                    "HumTargetGovOfficial": [],
                    "HumTargetMilitary": [],
                    "HumTargetPoliticalFigure": [],
                    "HumTargetLegal": [],
                    "HumTargetOthers": [],
                    "KIASingle": [],
                    "KIAPlural": [],
                    "KIAMultiple": [],
                    "WIASingle": [],
                    "WIAPlural": [],
                    "WIAMultiple": [],
                }
    return file_object


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
        return re.sub(regex, " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def get_tokens(s):
    if not s:
        return []
    return normalize_answer(s).split()


def compute_exact(a_gold, a_pred):
    return int(normalize_answer(a_gold) == normalize_answer(a_pred))


def compute_f1(a_gold, a_pred):
    gold_toks = get_tokens(a_gold)
    pred_toks = get_tokens(a_pred)
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
        return int(gold_toks == pred_toks)
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


# def read_golds_from_test_file(data_dir, tokenizer):
#     golds = OrderedDict()
#     doctexts_tokens = OrderedDict()
#     file_path = os.path.join(data_dir, "test.json")
#     with open(file_path, encoding="utf-8") as f:
#         for line in f:
#             line = json.loads(line)
#             docid = line["docid"]
#             # docid = int(line["docid"].split("-")[0][-1])*10000 + int(line["docid"].split("-")[-1]) # transform TST1-MUC3-0001 to int(0001)
#             doctext = line["doctext"]
#             if len(line["templates"]) > 0:
#                 extracts_raw = line["templates"][0]
#                 del extracts_raw['incident_type']

#             else:
#                 extracts_raw = {  "Location": [], "PerpInd": [], "PerpOrg": [], "PhysicalTarget": [], "Weapon": [], "HumTargetCivilian": [], "HumTargetGovOfficial": [], "HumTargetMilitary": [], "HumTargetPoliticalFigure": [], "HumTargetLegal": [], "HumTargetOthers": [], "KIASingle": [], "KIAPlural": [], "KIAMultiple": [], "WIASingle": [], "WIAPlural": [], "WIAMultiple": []}

#             extracts = OrderedDict()
#             for role, entitys_raw in extracts_raw.items():
#                 extracts[role] = []
#                 for entity_raw in entitys_raw:
#                     entity = []
#                     for mention_offset_pair in entity_raw:
#                         entity.append(mention_offset_pair[0])
#                     if entity:
#                         extracts[role].append(entity)
#             doctexts_tokens[docid] = tokenizer.tokenize(doctext)
#             golds[docid] = extracts
#     return doctexts_tokens, golds

# GTT Original
# def read_golds_from_test_file(data_dir, tokenizer, debug=False):
#     golds = OrderedDict()
#     doctexts_tokens = OrderedDict()
#     file_path = os.path.join(data_dir, "test.json")
#     with open(file_path, encoding="utf-8") as f:
#         example_cnt = 0
#         for line in f:
#             example_cnt += 1
#             if example_cnt > 3 and debug:
#                 break
#             line = json.loads(line)
#             # transform TST1-MUC3-0001 to int(0001)
#             docid = int(line["docid"].split("-")[0][-1]) * \
#                 10000 + int(line["docid"].split("-")[-1])
#             doctext, templates_raw = line["doctext"], line["templates"]

#             templates = []
#             for template_raw in templates_raw:
#                 template = OrderedDict()
#                 for role, value in template_raw.items():
#                     if role == "incident_type":
#                         template[role] = value
#                     else:
#                         template[role] = []
#                         for entity_raw in value:
#                             # first_mention_tokens = tokenizer.tokenize(entity[0][0])
#                             # start, end = find_sub_list(first_mention_tokens, doctext_tokens)
#                             # if start != -1 and end != -1:
#                             #     template[role].append([start, end])
#                             entity = []
#                             for mention_offset_pair in entity_raw:
#                                 entity.append(mention_offset_pair[0])
#                             if entity:
#                                 template[role].append(entity)

#                 if template not in templates:
#                     templates.append(template)

#             # for role, entitys_raw in extracts_raw.items():
#             #     extracts[role] = []
#             #     for entity_raw in entitys_raw:
#             #         entity = []
#             #         for mention_offset_pair in entity_raw:
#             #             entity.append(mention_offset_pair[0])
#             #         if entity:
#             #             extracts[role].append(entity)

#             doctexts_tokens[docid] = tokenizer.tokenize(doctext)
#             golds[docid] = templates
#     # import ipdb; ipdb.set_trace()
#     return doctexts_tokens, golds


def read_golds_from_test_file(data_dir, tokenizer, cfg, filename="test.json"):
    golds = OrderedDict()
    file_path = os.path.join(data_dir, filename)
    raw_gold_file = read_json(file_path)
    # raw_gold_incidents = {incident['docid']: incident
    #                       for incident in raw_gold_file}
    raw_gold_tokens = {
        incident["docid"]: tokenizer.tokenize(incident["doctext"])
        for incident in raw_gold_file
    }
    raw_gold_templates = {
        incident["docid"]: incident["templates"] for incident in raw_gold_file
    }

    default_template = OrderedDict()
    for key in cfg.role_map.keys():
        default_template[key] = [[]]

    for docid, raw_templates in raw_gold_templates.items():
        templates = []

        if len(raw_templates) > 0:
            for template_raw in raw_templates:
                template = OrderedDict()
                if len(template_raw.keys()) > 0:
                    incident_type = template_raw.pop("incident_type")
                    for role in cfg.role_map.keys():
                        value = template_raw[role]
                        template[role] = []
                        for entity_raw in value:
                            entity = []
                            for mention_offset_pair in entity_raw:
                                entity.append(mention_offset_pair[0])
                            if len(entity) > 0:
                                template[role].append(entity)
                            else:
                                template[role].append([[]])
                else:
                    template = default_template

                if template not in templates:
                    templates.append(template)

        else:
            templates.append(default_template)

        golds[docid] = templates[0]

    return raw_gold_tokens, golds

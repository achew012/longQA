import jsonlines
from collections import OrderedDict
import re
import json
import os


def write_json(filename, file_object):
    with open(filename, 'w') as file:
        file.write(json.dumps(file_object))


def to_jsonl(filename: str, file_obj):
    resultfile = open(filename, 'wb')
    writer = jsonlines.Writer(resultfile)
    writer.write_all(file_obj)


def read_json(jsonfile):
    with open(jsonfile, 'rb') as file:
        file_object = [json.loads(sample) for sample in file]
    return file_object


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
        return re.sub(regex, ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

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


def read_golds_from_test_file(data_dir, tokenizer, cfg, filename="test.json"):
    golds = OrderedDict()
    file_path = os.path.join(data_dir, filename)
    raw_gold_file = read_json(file_path)
    # raw_gold_incidents = {incident['docid']: incident
    #                       for incident in raw_gold_file}
    raw_gold_tokens = {incident['docid']: tokenizer.tokenize(incident["doctext"])
                       for incident in raw_gold_file}
    raw_gold_templates = {incident['docid']: incident["templates"]
                          for incident in raw_gold_file}

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
                        if not value:
                            template[role] = [[]]
                        for entity_raw in value:
                            # first_mention_tokens = tokenizer.tokenize(entity[0][0])
                            # start, end = find_sub_list(first_mention_tokens, doctext_tokens)
                            # if start != -1 and end != -1:
                            #     template[role].append([start, end])
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

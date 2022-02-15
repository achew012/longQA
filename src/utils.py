import jsonlines
from collections import OrderedDict
import re
import json
import os


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


def read_golds_from_test_file(data_dir, tokenizer):
    golds = OrderedDict()
    doctexts_tokens = OrderedDict()
    file_path = os.path.join(data_dir, "test.json")
    with open(file_path, encoding="utf-8") as f:
        for line in f:
            line = json.loads(line)
            docid = line["docid"]
            # docid = int(line["docid"].split("-")[0][-1])*10000 + int(line["docid"].split("-")[-1]) # transform TST1-MUC3-0001 to int(0001)
            doctext, extracts_raw = line["doctext"], line["extracts"]

            extracts = OrderedDict()
            for role, entitys_raw in extracts_raw.items():
                extracts[role] = []
                for entity_raw in entitys_raw:
                    entity = []
                    for mention_offset_pair in entity_raw:
                        entity.append(mention_offset_pair[0])
                    if entity:
                        extracts[role].append(entity)
            doctexts_tokens[docid] = tokenizer.tokenize(doctext)
            golds[docid] = extracts
    return doctexts_tokens, golds

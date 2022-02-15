import torch
from torch.utils.data import Dataset
import ipdb

role_map = {
    'PerpOrg': 'perpetrator organizations',
    'PerpInd': 'perpetrator individuals',
    'Victim': 'victims',
    'Target': 'targets',
    'Weapon': 'weapons'
}


class NERDataset(Dataset):
    # doc_list
    # extracted_list as template
    def __init__(self, dataset, tokenizer, cfg):
        self.tokenizer = tokenizer

        self.processed_dataset = {
            "docid": [],
            "context": [],
            "input_ids": [],
            "attention_mask": [],
            "qns": [],
            "gold_mentions": [],
            "start": [],
            "end": []
        }

        overflow_count = 0

        for doc in dataset:
            docid = doc["docid"]
            # self.tokenizer.decode(self.tokenizer.encode(doc["doctext"]))
            context = doc["doctext"]

            # Only take the 1st label of each role
            qns_ans = [["who are the {} entities?".format(role_map[key].lower()), doc["extracts"][key][0][0][1] if len(doc["extracts"][key]) > 0 else 0, doc["extracts"][key][0][0][1]+len(
                doc["extracts"][key][0][0][0]) if len(doc["extracts"][key]) > 0 else 0, doc["extracts"][key][0][0][0] if len(doc["extracts"][key]) > 0 else ""] for key in doc["extracts"].keys()]

            # expand on all labels in each role
            # qns_ans = [["who are the {} entities?".format(role_map[key].lower()), mention[1] if len(mention)>0 else 0, mention[1]+len(mention[0]) if len(mention)>0 else 0, mention[0] if len(mention)>0 else ""] for key in doc["extracts"].keys() for cluster in doc["extracts"][key] for mention in cluster]

            # labels = [self.labels2idx["O"]]*(self.tokenizer.model_max_length)
            # length_of_sequence = len(self.tokenizer.tokenize(context)) if len(self.tokenizer.tokenize(context))<=len(labels) else len(labels)
            # labels[:length_of_sequence] = [self.labels2idx["O"]]*length_of_sequence

            for qns, ans_char_start, ans_char_end, mention in qns_ans:
                context_encodings = self.tokenizer(qns, context, padding="max_length", truncation=True,
                                                   max_length=cfg.max_input_len, return_offsets_mapping=True, return_tensors="pt")
                sequence_ids = context_encodings.sequence_ids()

                # Detect if the answer is out of the span (in which case this feature is labeled with the CLS index).
                qns_offset = sequence_ids.index(1)-1
                pad_start_idx = sequence_ids[sequence_ids.index(
                    1):].index(None)
                offsets_wo_pad = context_encodings["offset_mapping"][0][qns_offset:pad_start_idx]

                # if char indices more than end idx in last word span
                if ans_char_end > offsets_wo_pad[-1][1] or ans_char_start > offsets_wo_pad[-1][1]:
                    overflow_count += 1
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
                # print("mention: ", mention)

                # if token_span!=[0,0]:
                #     labels[token_span[0]:token_span[0]+1] = [self.labels2idx["B-"+qns]]
                #     if len(labels[token_span[0]+1:token_span[1]+1])>0 and (token_span[1]-token_span[0])>0:
                #         labels[token_span[0]+1:token_span[1]+1] = [self.labels2idx["I-"+qns]]*(token_span[1]-token_span[0])

                self.processed_dataset["docid"].append(docid)
                self.processed_dataset["context"].append(context)
                self.processed_dataset["input_ids"].append(
                    context_encodings["input_ids"].squeeze(0))
                self.processed_dataset["attention_mask"].append(
                    context_encodings["attention_mask"].squeeze(0))
                self.processed_dataset["qns"].append(docid)
                self.processed_dataset["gold_mentions"].append(mention)
                self.processed_dataset["start"].append(
                    token_span[0]+qns_offset)
                self.processed_dataset["end"].append(
                    token_span[1]+qns_offset+1)

            # self.processed_dataset["labels"].append(torch.tensor(labels))
            # ipdb.set_trace()

        print("OVERFLOW COUNT: ", overflow_count)

    def __len__(self):
        """Returns length of the dataset"""
        return len(self.processed_dataset["docid"])

    def __getitem__(self, idx):
        """Gets an example from the dataset. The input and output are tokenized and limited to a certain seqlen."""
        # item = {key: val[idx] for key, val in self.processed_dataset["encodings"].items()}
        item = {}
        item['input_ids'] = self.processed_dataset["input_ids"][idx]
        item['attention_mask'] = self.processed_dataset["attention_mask"][idx]
        item['docid'] = self.processed_dataset["docid"][idx]
        item['gold_mentions'] = self.processed_dataset["gold_mentions"][idx]
        item['start'] = torch.tensor(self.processed_dataset["start"])[idx]
        item['end'] = torch.tensor(self.processed_dataset["end"])[idx]
        return item

    @staticmethod
    def collate_fn(batch):
        """
        Groups multiple examples into one batch with padding and tensorization.
        The collate function is called by PyTorch DataLoader
        """

        docids = [ex['docid'] for ex in batch]
        gold_mentions = [ex['gold_mentions'] for ex in batch]
        input_ids = torch.stack([ex['input_ids'] for ex in batch])
        attention_mask = torch.stack([ex['attention_mask'] for ex in batch])
        start = torch.stack([ex['start'] for ex in batch])
        end = torch.stack([ex['end'] for ex in batch])

        return {
            'docid': docids,
            'gold_mentions': gold_mentions,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'start_positions': start,
            'end_positions': end,
        }

from typing import Iterable, Optional, List
import torch
from random import sample
from rex.utils.logging import logger
from rex.utils.progress_bar import tqdm
from rex.data.label_encoder import LabelEncoder
from rex.data.transforms.base import TransformBase
from rex.utils.io import load_json
from transformers import BertTokenizerFast


class TextClassificationTransform(TransformBase):
    """Cached data transform for classification task."""

    def __init__(self, tokenizer: BertTokenizerFast, max_seq_len: int, label2id_filepath: str, use_partial:bool) -> None:
        super().__init__(max_seq_len)

        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        label2id = load_json(label2id_filepath)
        self.label_encoder = LabelEncoder(initial_dict = label2id)
        self.use_partial = use_partial

    def tokenize(self, text: str) -> dict:
        tokenized = self.tokenizer.encode_plus(
            text=text,
            add_special_tokens=True,
            padding="max_length",
            max_length=self.max_seq_len,
            truncation=True,
            return_attention_mask=True,
            return_token_type_ids=True,
        )
        return tokenized

    def collate_fn(self, data):
        final_data = {
            "textId": [],
            "text": [],
            "input_ids": [],
            "attention_mask": [],
            "token_type_ids": [],
            "labels": [],
        }
        for d in data:
            for key in final_data:
                if key in d:
                    final_data[key].append(d[key])

        final_data["input_ids"] = torch.tensor(final_data["input_ids"], dtype=torch.long)
        final_data["attention_mask"] = torch.tensor(
            final_data["attention_mask"], dtype=torch.long
        )
        final_data["token_type_ids"] = torch.tensor(
            final_data["token_type_ids"], dtype=torch.long
        )
        if final_data["labels"][0] != None:
            final_data["labels"] = torch.tensor(final_data["labels"], dtype=torch.long)
        else:
            final_data["labels"] = None
        return final_data

    def transform(
        self,
        dataset: Iterable,
        desc: Optional[str] = "Transform",
        debug: Optional[bool] = False,
        **kwargs,
    ) -> List[dict]:
        final_data = []
        if debug:
            dataset = dataset[:50]
        transform_loader = tqdm(dataset, desc=desc)

        num_tot_ins = 0
        for data in transform_loader:
            text = data["text"]
            tokenized = self.tokenize(text)
            input_ids = tokenized["input_ids"]
            attention_mask = tokenized["attention_mask"]
            token_type_ids = tokenized["token_type_ids"]
            
            labels = set()
            for label in data["labels"]:
                label_id = self.label_encoder.update_encode_one(label)
                labels.add(label_id)
            labels = list(labels)
            new_labels = self.label_encoder.convert_to_multi_hot(labels)
            if self.use_partial:
                partial_labels = data['partial_labels']
                for label in partial_labels:
                    label_id = self.label_encoder.update_encode_one(label)
                    new_labels[label_id] = 2 

            ins = {
                "textId": data["textId"],
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "token_type_ids": token_type_ids,
                "labels": new_labels,
                "text": text,
            }
            final_data.append(ins)
            num_tot_ins += 1

        logger.info(transform_loader)
        logger.info(f"#Total Ins: {num_tot_ins}")

        return final_data

    def eval_transform(
        self,
        dataset: Iterable,
        desc: Optional[str] = "Transform",
        debug: Optional[bool] = False,
        **kwargs,
    ) -> List[dict]:
        final_data = []
        if debug:
            dataset = dataset[:50]
        transform_loader = tqdm(dataset, desc=desc)

        num_tot_ins = 0
        for data in transform_loader:
            text = data["text"]
            tokenized = self.tokenize(text)
            input_ids = tokenized["input_ids"]
            attention_mask = tokenized["attention_mask"]
            token_type_ids = tokenized["token_type_ids"]
            
            labels = set()
            for label in data["labels"]:
                label_id = self.label_encoder.update_encode_one(label)
                labels.add(label_id)
            labels = list(labels)
            new_labels = self.label_encoder.convert_to_multi_hot(labels)

            ins = {
                "textId": data["textId"],
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "token_type_ids": token_type_ids,
                "labels": new_labels,
                "text": text,
                'label_types': None,
            }
            final_data.append(ins)
            num_tot_ins += 1

        logger.info(transform_loader)
        logger.info(f"#Total Ins: {num_tot_ins}")

        return final_data

    def predict_transform(self, data: dict):
        """
        Args:
            data:
                {
                    "textId": "textId",
                    "text": "text",
                }
        """
        text = data["text"]
        tokenized = self.tokenize(text)
        input_ids = tokenized["input_ids"]
        attention_mask = tokenized["attention_mask"]
        token_type_ids = tokenized["token_type_ids"]

        obj = {
            "textId": data["textId"],
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            "text": text,
            "labels": None,
            'label_types': None,
        }
        return obj


class LACOTransform(TransformBase):
    def __init__(self, tokenizer: BertTokenizerFast, max_seq_len: int, label2id_filepath: str, use_partial:bool) -> None:
        super().__init__(max_seq_len)
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        label2id = load_json(label2id_filepath)
        self.label_encoder = LabelEncoder(initial_dict = label2id)
        self.use_partial = use_partial
        # use [label1] --- [labeln] to replace label
        special_tokens = []
        for i in range(len(label2id)):
            special_tokens.append(f"[label{i}]")
        self.tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
            
    def encode_text_labels_pad(self, text: str):
        text = text[: self.max_seq_len - 3 - self.label_encoder.num_tags]
        pad_len = self.max_seq_len - 3 - len(text) - self.label_encoder.num_tags
        labels = []
        for i in range(self.label_encoder.num_tags):
            labels.append(f"[label{i}]")
        tokens = (
            ["[CLS]"]
            + list(text)
            + ["[SEP]"]
            + labels
            + ["[SEP]"]
            + ["[PAD]"] * pad_len
        )

        token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        mask = [1] + [2] * len(text) + [3] + [4] * self.label_encoder.num_tags + [5] + [0] * pad_len
        mask = torch.tensor(mask, dtype=torch.long)
        attention_mask = mask.gt(0).float()
        token_type_ids = mask.gt(3).long()
        attention_mask = attention_mask.tolist()
        token_type_ids = token_type_ids.tolist()
        return token_ids, attention_mask, token_type_ids

    def collate_fn(self, data):
        final_data = {
            "textId": [],
            "text": [],
            "input_ids": [],
            "attention_mask": [],
            "token_type_ids": [],
            "labels": [],
        }
        for d in data:
            for key in final_data:
                if key in d:
                    final_data[key].append(d[key])

        final_data["input_ids"] = torch.tensor(final_data["input_ids"], dtype=torch.long)
        final_data["attention_mask"] = torch.tensor(
            final_data["attention_mask"], dtype=torch.long
        )
        final_data["token_type_ids"] = torch.tensor(
            final_data["token_type_ids"], dtype=torch.long
        )
        if final_data["labels"][0] != None:
            final_data["labels"] = torch.tensor(final_data["labels"], dtype=torch.long)
        else:
            final_data["labels"] = None
        return final_data

    def transform(
        self,
        dataset: Iterable,
        desc: Optional[str] = "Transform",
        debug: Optional[bool] = False,
        **kwargs,
    ) -> List[dict]:
        final_data = []
        if debug:
            dataset = dataset[:50]
        transform_loader = tqdm(dataset, desc=desc)

        num_tot_ins = 0
        for data in transform_loader:
            text = data["text"]
            input_ids, attention_mask, token_type_ids = self.encode_text_labels_pad(text)
            
            labels = set()
            for label in data["labels"]:
                label_id = self.label_encoder.update_encode_one(label)
                labels.add(label_id)
            labels = list(labels)
            new_labels = self.label_encoder.convert_to_multi_hot(labels)
            if self.use_partial:
                partial_labels = data['partial_labels']
                for label in partial_labels:
                    label_id = self.label_encoder.update_encode_one(label)
                    new_labels[label_id] = 2 

            ins = {
                "textId": data["textId"],
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "token_type_ids": token_type_ids,
                "labels": new_labels,
                "text": text,
            }
            final_data.append(ins)
            num_tot_ins += 1

        logger.info(transform_loader)
        logger.info(f"#Total Ins: {num_tot_ins}")

        return final_data

    def eval_transform(
        self,
        dataset: Iterable,
        desc: Optional[str] = "Transform",
        debug: Optional[bool] = False,
        **kwargs,
    ) -> List[dict]:
        final_data = []
        if debug:
            dataset = dataset[:50]
        transform_loader = tqdm(dataset, desc=desc)

        num_tot_ins = 0
        for data in transform_loader:
            text = data["text"]
            input_ids, attention_mask, token_type_ids = self.encode_text_labels_pad(text)
            
            labels = set()
            for label in data["labels"]:
                label_id = self.label_encoder.update_encode_one(label)
                labels.add(label_id)
            labels = list(labels)
            new_labels = self.label_encoder.convert_to_multi_hot(labels)

            ins = {
                "textId": data["textId"],
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "token_type_ids": token_type_ids,
                "labels": new_labels,
                "text": text,
                'label_types': None,
            }
            final_data.append(ins)
            num_tot_ins += 1

        logger.info(transform_loader)
        logger.info(f"#Total Ins: {num_tot_ins}")

        return final_data

    def predict_transform(self, data: dict):
        """
        Args:
            data:
                {
                    "textId": "textId",
                    "text": "text",
                }
        """
        text = data["text"]
        input_ids, attention_mask, token_type_ids = self.encode_text_labels_pad(text)

        obj = {
            "textId": data["textId"],
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            "text": text,
            "labels": None,
            'label_types': None,
        }
        return obj


class HTTNTransform(TransformBase):
    def __init__(self, tokenizer: BertTokenizerFast, max_seq_len: int, label2id_filepath: str, num_head_labels:int) -> None:
        super().__init__(max_seq_len)
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.label2id = load_json(label2id_filepath)
        self.label_encoder = LabelEncoder(initial_dict = self.label2id)
        self.num_head_labels = num_head_labels
        self.head_label2id = {}
        for key,value in list(self.label2id.items())[:num_head_labels]:
            self.head_label2id[key] = value
        self.head_label_encoder = LabelEncoder(initial_dict = self.head_label2id)
        
    def tokenize(self, text: str) -> dict:
        tokenized = self.tokenizer.encode_plus(
            text=text,
            add_special_tokens=True,
            padding="max_length",
            max_length=self.max_seq_len,
            truncation=True,
            return_attention_mask=True,
            return_token_type_ids=True,
        )
        return tokenized

    def collate_fn(self, data):
        final_data = {
            "textId": [],
            "text": [],
            "input_ids": [],
            "attention_mask": [],
            "token_type_ids": [],
            "labels": [],
        }
        for d in data:
            for key in final_data:
                if key in d:
                    final_data[key].append(d[key])

        final_data["input_ids"] = torch.tensor(final_data["input_ids"], dtype=torch.long)
        final_data["attention_mask"] = torch.tensor(
            final_data["attention_mask"], dtype=torch.long
        )
        final_data["token_type_ids"] = torch.tensor(
            final_data["token_type_ids"], dtype=torch.long
        )
        if final_data["labels"][0] != None:
            final_data["labels"] = torch.tensor(final_data["labels"], dtype=torch.long)

        else:
            final_data["labels"] = None
        return final_data

    def transform(
        self,
        dataset: Iterable,
        desc: Optional[str] = "Transform",
        debug: Optional[bool] = False,
        **kwargs,
    ) -> List[dict]:
        final_data = []
        if debug:
            dataset = dataset[:50]
        transform_loader = tqdm(dataset, desc=desc)

        num_tot_ins = 0
        for data in transform_loader:
            text = data["text"]
            tokenized = self.tokenize(text)
            input_ids = tokenized["input_ids"]
            attention_mask = tokenized["attention_mask"]
            token_type_ids = tokenized["token_type_ids"]
            
            labels = set()
            for label in data["labels"]:
                label_id = self.label_encoder.update_encode_one(label)
                labels.add(label_id)
            labels = list(labels)
            new_labels = self.label_encoder.convert_to_multi_hot(labels)

            ins = {
                "textId": data["textId"],
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "token_type_ids": token_type_ids,
                "labels": new_labels,
                "text": text,
            }
            final_data.append(ins)
            num_tot_ins += 1

        logger.info(transform_loader)
        logger.info(f"#Total Ins: {num_tot_ins}")

        return final_data

    def head_transform(
        self,
        dataset: Iterable,
        desc: Optional[str] = "Transform",
        debug: Optional[bool] = False,
        **kwargs,
    ) -> List[dict]:
        final_data = []
        if debug:
            dataset = dataset[:50]
        transform_loader = tqdm(dataset, desc=desc)

        num_tot_ins = 0
        for data in transform_loader:            
            labels = set()
            for label in data["labels"]:
                if label in self.head_label2id:
                    label_id = self.head_label_encoder.update_encode_one(label)
                    labels.add(label_id)
            if len(labels) == 0:
                continue
            labels = list(labels)
            new_labels = self.head_label_encoder.convert_to_multi_hot(labels)
            text = data["text"]
            tokenized = self.tokenize(text)
            input_ids = tokenized["input_ids"]
            attention_mask = tokenized["attention_mask"]
            token_type_ids = tokenized["token_type_ids"]
            ins = {
                "textId": data["textId"],
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "token_type_ids": token_type_ids,
                "labels": new_labels,
                "text": text,
            }
            final_data.append(ins)
            num_tot_ins += 1

        logger.info(transform_loader)
        logger.info(f"#Total Ins: {num_tot_ins}")

        return final_data

    def predict_transform(self, data: dict):
        """
        Args:
            data:
                {
                    "textId": "textId",
                    "text": "text",
                }
        """
        text = data["text"]
        tokenized = self.tokenize(text)
        input_ids = tokenized["input_ids"]
        attention_mask = tokenized["attention_mask"]
        token_type_ids = tokenized["token_type_ids"]

        obj = {
            "textId": data["textId"],
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            "text": text,
            "labels": None,
        }
        return obj

    def sample_head_tail(self, sample_pre_num:int, sample_times:int, head_tail:str, datas:list):
        if head_tail == 'head':
            labels = self.head_label2id.keys()
        elif head_tail == 'tail':
            labels = list(self.label2id.keys())[self.num_head_labels:]
        else:
            raise ValueError(f'error head_tail {head_tail}')
        label2datas = {label:[] for label in labels}
        for data in datas:
            for label in data['labels']:
                if label in label2datas:
                    label2datas[label].append(data)
        result = []
        for _ in range(sample_times):
            sample_label2data = {}
            for label,label_datas in label2datas.items():
                sample_label2data[label] = sample(label_datas, sample_pre_num)
            result.append(sample_label2data)
        return result
    
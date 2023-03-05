import argparse
import logging
import os
from typing import *

import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import TensorDataset
from tqdm import tqdm
from transformers import BertTokenizerFast

from utils.processor_base import BaseDataProcessor
from utils.common import init_logger, write_txt, read_txt
from utils.data_base import BaseExample, BaseFeature

init_logger()
logger = logging.getLogger(__name__)


def generate_train_test_dev_corpus(args: argparse):
    """将给定的数据集划分为训练集,验证集和测试集,并保存。
    :param args:
    :return:
    """
    with open(os.path.join(args.data_dir, args.file_name), mode="r", encoding="utf-8") as f_obj:
        lines = [eval(line) for line in f_obj.readlines()]
    label_info = get_label_info(lines)
    train_data, test_data = train_test_split(np.array(lines), train_size=0.6, shuffle=True)
    test_data, dev_data = train_test_split(np.array(lines), train_size=0.5, shuffle=True)
    write_txt(train_data, os.path.join(args.data_dir, args.train_file_name))
    write_txt(test_data, os.path.join(args.data_dir, args.test_file_name))
    write_txt(dev_data, os.path.join(args.data_dir, args.dev_file_name))
    write_txt(label_info, os.path.join(args.data_dir, args.label_file_name))


class Example(BaseExample):
    def __init__(self, guid: str, text: str, entity_list: List[Dict]):
        """自定义数据结构.
        :param guid: 文本的唯一标识.
        :param text: 文本信息.
        :param entity_list: 实体列表.
        """
        super().__init__(guid, text)
        self.entity_list = entity_list


class Feature(BaseFeature):
    def __init__(self, input_ids: Optional[List],
                 attention_mask: Optional[List],
                 token_type_ids: Optional[List],
                 label_ids: Optional[List]):
        """自定义模型输入数据类型.
        :param input_ids: bert编码之后的token id列表
        :param attention_mask: bert编码之后的attention mask列表.
        :param token_type_ids: bert编码之后的token type 列表.
        :param label_ids: 标签列表.
        """
        super().__init__(input_ids, attention_mask, token_type_ids)
        self.label_ids = label_ids


def get_label_info(lines: List[Dict[str, Any]]) -> List[str]:
    """获取语料中的实体类型列表.
    :return:
    """
    entity_type_set = set()
    for line in lines:
        entity_list = line.get("entity_list")
        for entity in entity_list:
            entity_type_set.add(entity.get("entity_type"))
    return list(entity_type_set)


def get_label(args, encoding_type: str = "BIOES") -> Dict[str, int]:
    """根据语料中的实体类型生成标签.
    :param args:
    :param encoding_type: 编码类型,默认是BIOES编码方式.
    :return: 经过编码之后的字典列表.
    """
    entity_type_list = read_txt(os.path.join(args.data_dir, args.label_file_name))
    label_mapping_dict = {"O": 0}
    for entity in entity_type_list:
        for ch in encoding_type:
            if ch == "O":
                continue
            label_mapping_dict[f"{ch}-{entity}"] = len(label_mapping_dict)
    return label_mapping_dict


def encoder_label(entity_list: List[Dict], offset_mapping: List[Tuple], max_seq_len: int, mapping_dict: Dict) \
        -> List[int]:
    """对输入样本的标签进行指定类型的编码格式.
    :param entity_list:
    :param offset_mapping:
    :param max_seq_len:
    :param mapping_dict:
    :return:
    """
    label_ids = [0] * max_seq_len
    for entity in entity_list:
        entity_type = entity.get("entity_type")
        if not entity_type:
            continue
        start, end = entity["entity_index"]["begin"], entity["entity_index"]["end"]
        if end - start == 1:
            label_ids[start + 1] = mapping_dict[f"S-{entity_type}"]
            continue
        flag = False
        for idx, (s, e) in enumerate(offset_mapping[1:], 1):
            if s == start:
                label_ids[idx] = mapping_dict[f"B-{entity_type}"]
                flag = True
            elif flag and e < end:
                label_ids[idx] = mapping_dict[f"I-{entity_type}"]
            elif flag and e == end:
                label_ids[idx] = mapping_dict[f"E-{entity_type}"]
                break

    return label_ids


class PeopleDailyDataProcessor(BaseDataProcessor):
    """对不同的数据处理时,会有不同的处理方法."""

    def __init__(self, args):
        super().__init__(args)
        self.max_seq_len = args.max_seq_len

    @classmethod
    def _read_corpus(cls, input_file: str, quote_char: Optional[str] = None) -> list:
        with open(input_file, mode="r", encoding="utf-8") as f_obj:
            lines = f_obj.readlines()
        return [eval(line) for line in lines]

    def _create_examples(self, lines: list, set_type: str):
        example_list = []
        for idx, line in tqdm(enumerate(lines)):
            guid = f"{set_type}-{idx}"
            text = line.get("text")
            entity_list = line.get("entity_list")
            if len(text) + 2 > self.max_seq_len or not entity_list:
                continue
            example_list.append(Example(guid=guid, text=text, entity_list=entity_list))
        return example_list

    @staticmethod
    def convert_examples_to_features(
            examples,
            max_seq_len,
            tokenizer,
            cls_token="[CLS]",
            cls_token_segment_id=0,
            sep_token="[SEP]",
            pad_token=0,
            pad_token_segment_id=0,
            sequence_a_segment_id=0,
            add_sep_token=False,
            mask_padding_with_zero=True
    ):

        features = []
        for (ex_index, example) in tqdm(enumerate(examples)):
            if ex_index % 1000 == 0:
                logger.info("Writing example %d of %d" % (ex_index, len(examples)))
            tokens = tokenizer(example.text, return_offsets_mapping=True, padding="max_length", max_length=max_seq_len)
            to = tokenizer.convert_ids_to_tokens(tokens.input_ids)
            input_ids = tokens["input_ids"]
            attention_mask = tokens["attention_mask"]
            token_type_ids = tokens["token_type_ids"]
            offset_mapping = tokens["offset_mapping"]
            label_ids = encoder_label(example.entity_list, offset_mapping, max_seq_len, mapping_dict)
            assert len(input_ids) == max_seq_len, "Error with input length {} vs {}".format(len(input_ids), max_seq_len)
            assert len(attention_mask) == max_seq_len, "Error with attention mask length {} vs {}".format(
                len(attention_mask), max_seq_len)
            assert len(token_type_ids) == max_seq_len, "Error with token type length {} vs {}".format(
                len(token_type_ids), max_seq_len)

            if ex_index % 1000 == 0:
                logger.info("*** Example ***")
                logger.info("guid: %s" % example.a_guid)
                logger.info("tokens: %s" % " ".join([str(x) for x in to]))
                logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
                logger.info("attention_mask: %s" % " ".join([str(x) for x in attention_mask]))
                logger.info("token_type_ids: %s" % " ".join([str(x) for x in token_type_ids]))
                logger.info("label: %s " % " ".join([str(x) for x in label_ids]))

            features.append(
                Feature(
                    input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
                    label_ids=label_ids))
        return features

    @classmethod
    def load_and_cache_examples(cls, args, tokenizer, mode):
        processor = cls(args)
        label2id = get_label(args)

        cached_features_file = os.path.join(
            args.data_dir,
            "cached_{}_{}_{}_{}".format(
                mode, args.task, list(filter(None, args.model_name_or_path.split("/"))).pop(), args.max_seq_len),
        )

        if os.path.exists(cached_features_file):
            # 如果路径存在,直接从该路径读取数据.
            logger.info("Loading features from cached file %s" % cached_features_file)
            features = torch.load(cached_features_file)
        else:
            # 如果不存在,需要生成数据并且保存到这个目录下.
            logger.info("Creating features from dataset file at %s" % args.data_dir)
            if mode == "train":
                examples = processor.get_examples("train", args.max_seq_len)
            elif mode == "dev":
                examples = processor.get_examples("dev", args.max_seq_len)
            elif mode == "test":
                examples = processor.get_examples("test", args.max_seq_len)
            else:
                raise Exception("For mode, Only train, dev, test is available")

            features = convert_examples_to_features(
                examples=examples, max_seq_len=args.max_seq_len, tokenizer=tokenizer, mapping_dict=label2id)
            logger.info("Saving features into cached file %s" % cached_features_file)
            # 保存数据.
            torch.save(features, cached_features_file)

        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
        all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_ids for f in features], dtype=torch.float)

        dataset = TensorDataset(
            all_input_ids,
            all_attention_mask,
            all_token_type_ids,
            all_label_ids
        )

        return dataset, label2id, id2label, label_num

    #
    # def read_original_corpus(self, mode: str) -> List[Dict]:
    #     """读取原始的语料库.
    #     :return:
    #     """
    #
    # def get_label_list(self) -> List[str]:
    #     """获取语料中的实体类型列表.
    #     :return:
    #     """
    #     entity_type_set = set()
    #     for line in self.dataset:
    #         entity_list = line.get("entity_list")
    #         for entity in entity_list:
    #             entity_type_set.add(entity.get("entity_type"))
    #     return list(entity_type_set)
    #
    # def create_examples(self, mode: str, max_seq_len: int):
    #     """生成样本列表.
    #     :param max_seq_len:
    #     :param mode: train test dev三种情况.
    #     :return:
    #     """
    #     example_list = []
    #     for idx, line in tqdm(enumerate(self.dataset)):
    #         guid = f"{mode}-{idx}"
    #         text = line.get("text")
    #         entity_list = line.get("entity_list")
    #         if len(text) + 2 > max_seq_len or not entity_list:
    #             continue
    #         example_list.append(Example(guid=guid, text=text, entity_list=entity_list))
    #     return example_list


def convert_examples_to_features(
        examples: List[Example],
        max_seq_len: int,
        tokenizer: BertTokenizerFast,
        mapping_dict: Dict
):
    features = []
    for (ex_index, example) in tqdm(enumerate(examples)):
        if ex_index % 1000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))
        tokens = tokenizer(example.text, return_offsets_mapping=True, padding="max_length", max_length=max_seq_len)
        to = tokenizer.convert_ids_to_tokens(tokens.input_ids)
        input_ids = tokens["input_ids"]
        attention_mask = tokens["attention_mask"]
        token_type_ids = tokens["token_type_ids"]
        offset_mapping = tokens["offset_mapping"]
        label_ids = encoder_label(example.entity_list, offset_mapping, max_seq_len, mapping_dict)
        assert len(input_ids) == max_seq_len, "Error with input length {} vs {}".format(len(input_ids), max_seq_len)
        assert len(attention_mask) == max_seq_len, "Error with attention mask length {} vs {}".format(
            len(attention_mask), max_seq_len)
        assert len(token_type_ids) == max_seq_len, "Error with token type length {} vs {}".format(
            len(token_type_ids), max_seq_len)

        if ex_index % 1000 == 0:
            logger.info("*** Example ***")
            logger.info("guid: %s" % example.a_guid)
            logger.info("tokens: %s" % " ".join([str(x) for x in to]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("attention_mask: %s" % " ".join([str(x) for x in attention_mask]))
            logger.info("token_type_ids: %s" % " ".join([str(x) for x in token_type_ids]))
            logger.info("label: %s " % " ".join([str(x) for x in label_ids]))

        features.append(
            Feature(
                input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, label_ids=label_ids))
    return features


def load_and_cache_examples(
        args, tokenizer: BertTokenizerFast,
        mode: str
) -> Tuple[TensorDataset, Dict[str, int], Dict[int, str], int]:
    processor = PeopleDailyDataProcessor(args)
    label_list = processor.get_label_list()
    label2id, id2label, label_num = get_label_info(label_list)

    cached_features_file = os.path.join(
        args.data_dir,
        "cached_{}_{}_{}_{}".format(
            mode, args.task, list(filter(None, args.model_name_or_path.split("/"))).pop(), args.max_seq_len),
    )

    if os.path.exists(cached_features_file):
        # 如果路径存在,直接从该路径读取数据.
        logger.info("Loading features from cached file %s" % cached_features_file)
        features = torch.load(cached_features_file)
    else:
        # 如果不存在,需要生成数据并且保存到这个目录下.
        logger.info("Creating features from dataset file at %s" % args.data_dir)
        if mode == "train":
            examples = processor.create_examples("train", args.max_seq_len)
        elif mode == "dev":
            examples = processor.create_examples("dev", args.max_seq_len)
        elif mode == "test":
            examples = processor.create_examples("test", args.max_seq_len)
        else:
            raise Exception("For mode, Only train, dev, test is available")

        features = convert_examples_to_features(
            examples=examples, max_seq_len=args.max_seq_len, tokenizer=tokenizer, mapping_dict=label2id)
        logger.info("Saving features into cached file %s" % cached_features_file)
        # 保存数据.
        torch.save(features, cached_features_file)

    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_ids for f in features], dtype=torch.float)

    dataset = TensorDataset(
        all_input_ids,
        all_attention_mask,
        all_token_type_ids,
        all_label_ids
    )

    return dataset, label2id, id2label, label_num

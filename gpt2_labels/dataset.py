import json
import logging
import os
from typing import *

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from tqdm import tqdm

logger = logging.getLogger(__name__)


class GPT2LabelDataSet(Dataset):
    def __init__(self, tokenizer, content_max_len: int, label_max_len: int, cache_data_dir: str,
                 data_set_name: str, path_file: str = None, is_overwrite: bool = False):
        """
        :param tokenizer: 分词器.
        :param content_max_len: 文本的最大长度,这个文本的长度是包含试题和试题标签的.
        :param label_max_len: 标题生成的最大长度.
        :param cache_data_dir: 缓存文件的保存路径.
        :param data_set_name: 数据集的名称.
        :param path_file: 文件路径.
        :param is_overwrite: 是否重新生成缓存文件.
        """
        self.tokenizer = tokenizer
        self.content_max_len = content_max_len
        self.label_max_len = label_max_len

        self.content_id = self.tokenizer.convert_tokens_to_ids("[Content]")
        self.label_id = self.tokenizer.convert_tokens_to_ids("[Label]")
        self.space_id = self.tokenizer.convert_tokens_to_ids("[Space]")
        cached_feature_file = os.path.join(cache_data_dir, "cached_{}_{}".format(data_set_name, content_max_len))

        if os.path.exists(cached_feature_file) and not is_overwrite:
            logger.info("已经存在缓存文件{},直接加载!".format(cached_feature_file))
            self.data_set = torch.load(cached_feature_file)["data_set"]
        else:
            logger.info("不存在缓存文件{},进行数据预处理操作!".format(cached_feature_file))
            self.data_set = self.load_data(path_file)
            logger.info("数据预处理操作完成,将处理后的数据存到{}中,作为缓存文件!".format(cached_feature_file))
            torch.save({"data_set": self.data_set}, cached_feature_file)

    def load_data(self, path_file: str):
        """加载原始数据,生成数据处理后的数据.
        :param path_file:
        :return:
        """
        data_set = []
        with open(path_file, mode="r", encoding="utf-8") as pf:
            data = json.load(pf)
            for _, sample in enumerate(tqdm(data, desc="iter", disable=False)):
                input_ids, token_type_ids = self.convert_feature(sample)
                data_set.append({"input_ids": input_ids, "token_type_ids": token_type_ids})
        return data_set

    def convert_feature(self, sample: Dict[str, str]) -> Tuple[List[int], List[int]]:
        """数据处理.
        :param sample: 包含题干和知识点,格式为{"content": content, "label": label}
        :return:
        """
        input_ids = []
        token_type_ids = []

        content_tokens = self.tokenizer.tokenize(sample["content"])
        # 这种分词方式并没有把标签的整体作为一个token进行预测.
        label_tokens = self.tokenizer.tokenize(sample["label"].replace(" ", "[Space]"))

        if len(label_tokens) > self.label_max_len:
            label_tokens = label_tokens[:self.label_max_len]
        # 三个特殊的token,分别是CLS,SEP和SEP.
        if len(content_tokens) > self.content_max_len - len(label_tokens) - 3:
            content_tokens = content_tokens[:self.content_max_len - len(label_tokens) - 3]

        input_ids.append(self.tokenizer.cls_token_id)
        token_type_ids.append(self.content_id)

        input_ids.extend(self.tokenizer.convert_tokens_to_ids(content_tokens))
        token_type_ids.extend([self.content_id] * len(content_tokens))

        input_ids.append(self.tokenizer.sep_token_id)
        token_type_ids.append(self.content_id)

        input_ids.extend(self.tokenizer.convert_tokens_to_ids(label_tokens))
        token_type_ids.extend([self.label_id] * len(label_tokens))

        input_ids.append(self.tokenizer.sep_token_id)
        token_type_ids.append(self.label_id)

        assert len(input_ids) == len(token_type_ids)
        assert len(input_ids) <= self.content_max_len
        return input_ids, token_type_ids

    def __len__(self):
        return len(self.data_set)

    def __getitem__(self, idx):
        return self.data_set[idx]


def collate_func(batch_data: List[Dict[str, str]]) -> Dict[Optional[str], Optional[torch.Tensor]]:
    """DataLoader所需的collate_fun函数,将数据处理成tensor形式.
    :param batch_data:
    :return:
    """
    batch_size = len(batch_data)
    if batch_size == 0:
        return {}
    input_ids_list, token_type_ids_list = [], []
    for instance in batch_data:
        input_ids_temp = instance["input_ids"]
        token_type_ids_temp = instance["token_type_ids"]
        input_ids_list.append(torch.tensor(input_ids_temp, dtype=torch.long))
        token_type_ids_list.append(torch.tensor(token_type_ids_temp, dtype=torch.long))
    return {
        "input_ids": pad_sequence(input_ids_list, batch_first=True, padding_value=0),
        "token_type_ids": pad_sequence(token_type_ids_list, batch_first=True, padding_value=0)
    }

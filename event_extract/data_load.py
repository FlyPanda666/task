import json
import os
import numpy as np
from torch.utils import data
from transformers import BertTokenizer

from event_extractor.predefine_token import NONE, PAD, UNK, CLS, SEP, TRIGGERS, ARGUMENTS
from event_extractor.utils_func import build_vocabulary
from utils.common.common_function import get_data_dir
from utils.common.my_logger import logger

pre_train_model = os.path.join(get_data_dir(), "torch_roberta_wwm")
all_triggers, trigger2idx, idx2trigger = build_vocabulary(TRIGGERS)
all_arguments, argument2idx, idx2argument = build_vocabulary(ARGUMENTS)
tokenizer = BertTokenizer.from_pretrained(pre_train_model, do_lower_case=False, never_split=(PAD, UNK, CLS, SEP))


class TrainDataset(data.Dataset):
    def __init__(self, file_dir: str, sentence_length: int):
        self.sentence_list, self.trigger_list, self.id_list, self.argument_list = [], [], [], []
        with open(file_dir, mode="r", encoding="utf-8") as reader:
            for line in reader.readlines():
                sample = json.loads(line)
                # id存储
                sample_id = sample.get("id")
                # 文本处理
                sentence = sample.get("text").replace(" ", "-")
                sentence = sentence.replace('\n', ',')
                sentence = sentence.replace('\u3000', '-')
                sentence = sentence.replace('\xa0', ',')
                sentence = sentence.replace('\ue627', ',')
                words = [word for word in sentence]
                if len(words) > sentence_length:
                    continue
                triggers = [NONE] * len(words)
                arguments = {"events": {}}
                for event_mention in sample.get("event_list"):
                    id_start = event_mention.get("trigger_start_index")
                    trigger_word = event_mention.get("trigger")
                    id_end = id_start + len(trigger_word)
                    event_type = event_mention.get("event_type").split("-")[-1]
                    for i in range(id_start, id_end):
                        if i == id_start:
                            triggers.append("B-{}".format(event_type))
                        else:
                            triggers.append("I-{}".format(event_type))
                    event_key = (id_start, id_end, event_type)
                    arguments["events"][event_key] = []
                    for argument in event_mention.get("arguments"):
                        role = argument.get("role")
                        argument_id_start = argument.get("argument_start_index")
                        argument_text = argument.get("argument")
                        argument_id_end = argument_id_start + len(argument_text)
                        arguments["events"][event_key].append((argument_id_start, argument_id_end, role))
                self.sentence_list.append([CLS] + words + [SEP])
                self.trigger_list.append(triggers)
                self.id_list.append(sample_id)
                self.argument_list.append(arguments)

    def __len__(self):
        return len(self.sentence_list)

    def __getitem__(self, idx):
        words, sample_id, triggers, arguments = \
            self.sentence_list[idx], self.id_list[idx], self.trigger_list[idx], self.argument_list[idx]

        tokens_x, is_heads = [], []
        for word in words:
            tokens = tokenizer.tokenize(word) if word not in [CLS, SEP] else [word]
            if len(tokens) != 1:
                logger().info(tokens)
                logger().info("This is not a single Chinese tokens!")
            tokens_xx = tokenizer.convert_tokens_to_ids(tokens)
            if word in [CLS, SEP]:
                is_head = [0]
            else:
                is_head = [1] + [0] * (len(tokens) - 1)
            tokens_x.extend(tokens_xx)
            is_heads.extend(is_head)
        triggers_y = [trigger2idx[t] for t in triggers]
        head_indexes = []
        for i in range(len(is_heads)):
            if is_heads[i]:
                head_indexes.append(i)
        sequence_length = len(tokens_x)
        mask = [1] * sequence_length

        return tokens_x, sample_id, triggers_y, arguments, sequence_length, head_indexes, mask, words, triggers

    def get_samples_weight(self):
        samples_weight = []
        for triggers in self.trigger_list:
            not_none = False
            for trigger in triggers:
                if trigger != NONE:
                    not_none = True
                    break
            if not_none:
                samples_weight.append(5.0)
            else:
                samples_weight.append(1.0)
        return np.array(samples_weight)


class TestDataset(data.Dataset):
    def __init__(self, file_dir: str, sentence_length: int):
        """
        :param file_dir:
        """
        self.sentence_list, self.id_list = [], []
        with open(file_dir, mode="r", encoding="utf-8") as reader:
            for line in reader.readlines():
                try:
                    sample = json.loads(line)
                except json.decoder.JSONDecodeError:
                    continue
                sample_id = sample.get("id")
                sentence = sample.get("text").replace(' ', '-')
                sentence = sentence.replace('\n', ',')
                sentence = sentence.replace('\u3000', '-')
                sentence = sentence.replace('\xa0', ',')
                sentence = sentence.replace('\ue627', ',')
                words = [word for word in sentence]
                if len(words) > sentence_length:
                    continue

                self.sentence_list.append([CLS] + words + [SEP])
                self.id_list.append(sample_id)

    def __len__(self):
        return len(self.sentence_list)

    def __getitem__(self, idx):
        words, id_list = self.sentence_list[idx], self.id_list[idx]
        tokens_x, is_heads = [], []
        for word in words:
            tokens = tokenizer.tokenize(word) if word not in [SEP, CLS] else [word]
            tokens__xx = tokenizer.convert_tokens_to_ids(tokens)
            if word in [CLS, SEP]:
                is_head = [0]
            else:
                is_head = [1] + [0] * (len(tokens) - 1)
            tokens_x.extend(tokens__xx)
            is_heads.extend(is_head)

        heads_indexes = []
        for i in range(len(is_heads)):
            if is_heads[i]:
                heads_indexes.append(i)
        sequence_length = len(tokens_x)
        mask = [1] * sequence_length

        return tokens_x, id_list, sequence_length, heads_indexes, mask, words


def train_pad(batch):
    """ 如果对处理好的数据没有额外的预处理可以不用使用该方法。
    直白来讲，就是拿到batch之后，对batch数据进行预处理，然后作为返回值返回。
    torch.utils.data.DataLoader是pytorch提供的数据加载类。
    初始化中提到collate_fn，官方解释为merges a list of samples to form a mini-batch of Tensor(s).
    Used when using batched loading from a map-style dataset.
    其实collate_fn可以理解为句柄、指针或者其他可以调用类(实现__call__函数)。函数的输入为list，list中的元素为欲处理的一系列样本。
    通过collate_fn函数可以对这些样本做进一步的处理，原则上返回值应当是一个有结构的batch。而DataLoader每次迭代的返回值就是collate_fn的返回值。

    :param batch:
    :return:
    """
    tokens_x_2d, sample_id, triggers_y_2d, arguments_2d, sequence_length_1d, head_indexes_2d, mask, words_2d, triggers_2d = list(map(list, zip(*batch)))
    max_length = np.array(sequence_length_1d).max()

    for i in range(len(tokens_x_2d)):
        tokens_x_2d[i] = tokens_x_2d[i] + [0] * (max_length - len(tokens_x_2d[i]))
        triggers_y_2d[i] = triggers_y_2d[i] + [trigger2idx[PAD]] * (max_length - len(triggers_y_2d[i]))
        head_indexes_2d[i] = head_indexes_2d[i] + [0] * (max_length - len(head_indexes_2d[i]))
        mask[i] = mask[i] + [0] * (max_length - len(mask[i]))

    return tokens_x_2d, sample_id, triggers_y_2d, arguments_2d, sequence_length_1d, head_indexes_2d, mask, words_2d, triggers_2d


def test_pad(batch):
    tokens_x_2d, sample_id, sequence_length_1d, head_indexes_2d, mask, words_2d = list(map(list, zip(*batch)))
    max_length = np.array(sequence_length_1d).max()

    for i in range(len(tokens_x_2d)):
        tokens_x_2d[i] = tokens_x_2d[i] + [0] * (max_length - len(tokens_x_2d[i]))
        head_indexes_2d[i] = head_indexes_2d[i] + [0] * (max_length - len(head_indexes_2d[i]))
        mask[i] = mask[i] + [0] * (max_length - len(mask[i]))

    return tokens_x_2d, sample_id, sequence_length_1d, head_indexes_2d, mask, words_2d


def test_train_class():
    td = TrainDataset("../../data_dir/event_extractor/train.json", sentence_length=500)
    for sample in td:
        logger().info(sample)
    # logger().info(td.get_samples_weight())


def test_test_class():
    td = TestDataset("test_data.json")
    logger().info(td.id_list)
    logger().info(td.sentence_list)
    logger().info(td[1])


if __name__ == '__main__':
    test_train_class()



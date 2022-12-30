import codecs
import json
import random
import re
from functools import partial
from multiprocessing import Pool, cpu_count
from typing import *

from tqdm import tqdm


def clean_weibo_title(title: str):
    """title 文本清洗.
    :param title:
    :return:
    """
    title = re.sub(r"#", "", title)
    title = re.sub(r"(\[{1,2})(.*?)(\]{1,2})", "", title)
    title = re.sub(r"\s+", " ", title)
    return title


def clean_weibo_content(content: str):
    """content 文本清洗.
    :param content:
    :return:
    """
    content = re.sub(r"(https|http)?:\/\/(\w|\.|\/|\?|\=|\&|\%)*\b", "", content)
    content = re.sub(r"\s+", " ", content)
    content = content.replace("\u200b", "")
    return content


def clean_data(sample: Tuple[str, str]) -> Dict[str, str]:
    """整体清洗函数，为了方便多线程使用.
    :param sample:
    :return:
    """
    (content, title) = sample
    sample = dict()
    sample["title"] = clean_weibo_title(title.strip())
    sample["content"] = clean_weibo_content(content.strip())
    return sample


def build_news_data(content_path: str, title_path: str, train_save_path: str, test_save_path: str,
                    filter_content_len: int = 100, filter_title_len: int = 2):
    """对数据进行清洗,构建训练集合测试集.训练数据的文本和标题是分开存储的.
    :param content_path: 文本路径.
    :param title_path: 文本对应的标题路径.
    :param train_save_path: 训练集文件的保存路径.
    :param test_save_path: 测试集文件的保存路径.
    :param filter_content_len: 考虑的最小文本长度.
    :param filter_title_len: 考虑的最小标题的长度.
    :return:
    """
    content_data = open(content_path, mode="r", encoding="utf-8")
    title_data = open(title_path, mode="r", encoding="utf-8")
    data = zip(content_data.readlines(), title_data.readlines())
    # 使用多进程处理数据.
    threads = min(8, cpu_count())
    with Pool(threads) as p:
        func = partial(clean_data)
        data = list(tqdm(p.imap(func, data, chunksize=8), desc="build data"))
    data_set = set()
    data_new = []
    for d in data:
        if d["content"] in data_set or len(d["content"]) < filter_content_len or len(d["title"]) < filter_title_len:
            continue
        else:
            data_set.add(d["content"])
            data_new.append(d)

    random.shuffle(data_new)
    train_data = data_new[:-3000]
    test_data = data_new[-3000:]
    with open(train_save_path, mode="w", encoding="utf-8") as tr_writer:
        tr_writer.write(json.dumps(train_data, indent=4, ensure_ascii=False))
    with open(test_save_path, mode="w", encoding="utf-8") as te_writer:
        te_writer.write(json.dumps(test_data, indent=4, ensure_ascii=False))


def read_raw_data(data_dir: str, text_dir: str, label_dir: str):
    """读取原始数据,然后将数据拆分为两个部分,其中的一部分用来存放文本,另一个部分用来存放文本对应的标签.
    :param data_dir: 原始数据的保存路径.
    :param text_dir: 文本路径.
    :param label_dir: 文本对应的标题路径.
    :return:
    """
    title_list = []
    content_list = []
    with codecs.open(data_dir, mode="r", encoding="utf-8") as reader:
        data = json.load(reader)
    for item in data:
        title_list.append(item["title"])
        content_list.append(item["content"])
    with open(text_dir, "w", encoding="utf-8") as te:
        for line in content_list:
            te.write(line + "\n")
    with open(label_dir, mode="w", encoding="utf-8") as la:
        for line in title_list:
            la.write(line + "\n")


if __name__ == '__main__':
    import os
    if not os.path.exists("data_dir"):
        os.makedirs("data_dir", exist_ok=True)
    raw_data_dir = "/Users/tal/Desktop/Book/weibo_data.json"
    content_path_dir = "data_dir/train_text.txt"
    title_path_dir = "data_dir/train_label.txt"
    train_save_path_dir = "data_dir/train_data.json"
    test_save_path_dir = "data_dir/test_data.json"
    read_raw_data(raw_data_dir, content_path_dir, title_path_dir)
    build_news_data(content_path_dir, title_path_dir, train_save_path_dir, test_save_path_dir)

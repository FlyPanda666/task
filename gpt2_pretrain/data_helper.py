import json

import pandas as pd


def convert_to_json(raw_data_path: str, json_file_path: str):
    """将存储在DataFrame中的数据生成json格式的文件.
    :param raw_data_path: 原始文件的文件路径.
    :param json_file_path: 处理结束之后保存的文件路径.
    :return:
    """
    data = pd.read_pickle(raw_data_path)
    text = data["text"].to_list()
    with open(json_file_path, mode="w", encoding="utf-8") as writer:
        json.dump(text, writer)


if __name__ == '__main__':
    raw_dir = "./data_dir/试题id和text.pickle"
    file_path = "./data_dir/试题题目和选项文本信息文本.json"
    convert_to_json(raw_dir, file_path)

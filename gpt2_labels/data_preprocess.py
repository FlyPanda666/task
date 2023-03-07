import pandas as pd
import json
import os
from sklearn.model_selection import train_test_split


def build_data(input_dir: str, output_dir: str = "./data_dir"):
    """构建模型的训练数据,测试数据和验证数据集.
    :param input_dir:
    :param output_dir:
    :return:
    """
    data = pd.read_pickle(input_dir)
    train, test = train_test_split(data, train_size=0.8)
    train, dev = train_test_split(train, train_size=0.8)
    train.to_pickle(os.path.join(output_dir, "train.pkl"))
    test.to_pickle(os.path.join(output_dir, "test.pkl"))
    dev.to_pickle(os.path.join(output_dir, "dev.pkl"))


def build_news_data(data_dir: str, mode: str):
    """处理原始数据,存储为JSON文件格式.
    :param data_dir: 原始数据存放的地址.
    :param mode: 数据的类型,分为训练数据,测试数据,验证数据.
    :return:
    """
    file = os.path.join(data_dir, mode+".pkl")
    df_data = pd.read_pickle(file)
    data_new = []
    for text, label in zip(df_data["text"], df_data["new_label_name"]):
        data_new.append({"label": " ".join(eval(label)), "content": text.strip()})
    print(f"{mode}数据的大小为{len(data_new)}")

    output_path = os.path.join(data_dir, mode + "_data_grade3.json")
    with open(output_path, mode="w", encoding="utf-8") as te_writer:
        te_writer.write(json.dumps(data_new, indent=4, ensure_ascii=False))


if __name__ == "__main__":
    build_data("../dev.pkl")
    read_data_dir = "./data_dir"
    for data_mode in ["train", "test", "dev"]:
        build_news_data(read_data_dir, mode=data_mode)

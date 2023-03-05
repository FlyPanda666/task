from transformers import BertTokenizer
import numpy as np
import torch
import config
from model import BertNER
from metrics import get_entities


def infer(in_sentence: str):
    tokenizer = BertTokenizer.from_pretrained("/Users/tal/Desktop/Document/torch_roberta_wwm/", do_lower_case=True)
    # line = list(in_sentence)
    line = [char for char in list(in_sentence) if char != " "]
    sentence = []
    words = []
    word_len = []
    batch_label_starts = []
    pred_tags = []
    for token in line:
        words.append(tokenizer.tokenize(token))
        word_len.append(len(token))
    words = ["[CLS]"] + [item for token in words for item in token]
    token_start_index = 1 + np.cumsum([0] + word_len)
    print(1, len(words))
    print(2, len(token_start_index))
    sentence.append((tokenizer.convert_tokens_to_ids(words), token_start_index))
    print(3, sentence)
    label_starts = np.ones((1, len(words)))
    batch_label_starts.append(label_starts)
    batch_label_starts = torch.tensor(batch_label_starts, dtype=torch.long)
    batch_data = torch.tensor(sentence[0], dtype=torch.long)
    model = BertNER.from_pretrained(config.model_dir)
    batch_output = model((batch_data, batch_label_starts))[0]
    batch_output = model.crf.decode(batch_output[0])
    pred_tags.extend([[config.id2label.get(idx) for idx in indices] for indices in batch_output])
    res = [element[-1] for element in pred_tags]
    for word, tag in zip(words, res):
        print(word + "->" + tag)

    chuck = get_entities(res)
    print(chuck)
    ans = {"keyword": [], "aspect": [], "influence": []}
    res = []
    for item in chuck:
        res.append({item[0]: "".join(words[item[1]: item[2] + 1])})
        ans[item[0]].extend(words[item[1]: item[2] + 1])
    print(joint_words(ans))
    print(res)


def joint_words(parameters: dict):
    for key, value in parameters.items():
        parameters[key] = "".join(value)
    return parameters


def predict(input_sentence, model, tokenizer):
    line = [char for char in list(input_sentence) if char != " "]
    sentence = []
    words = []
    word_len = []
    batch_label_starts = []
    pred_tags = []
    for token in line:
        words.append(tokenizer.tokenize(token))
        word_len.append(len(token))
    words = ["[CLS]"] + [item for token in words for item in token]
    token_start_index = 1 + np.cumsum([0] + word_len)

    sentence.append((tokenizer.convert_tokens_to_ids(words), token_start_index))
    label_starts = np.ones((1, len(words)))
    batch_label_starts.append(label_starts)
    batch_label_starts = torch.tensor(batch_label_starts, dtype=torch.long)
    batch_data = torch.tensor(sentence[0], dtype=torch.long)
    batch_output = model((batch_data, batch_label_starts))[0]
    batch_output = model.crf.decode(batch_output[0])
    pred_tags.extend([[config.id2label.get(idx) for idx in indices] for indices in batch_output])
    res = [element[-1] for element in pred_tags]
    # for word, tag in zip(words, res):
    #     print(word + "->" + tag)

    chuck = get_entities(res)
    ans = {"keyword": [], "aspect": [], "influence": []}
    res = []
    for item in chuck:
        res.append({item[0]: "".join(words[item[1]: item[2] + 1])})
    # if len(res) == 3:
    return res


def read_data():
    with open("/Users/tal/Desktop/CLUENER2020的副本/BERT-CRF/data/clue/new_data.txt", mode="r", encoding="utf-8") as f:
        for line in f.readlines():
            yield line.strip()


if __name__ == '__main__':
    model = BertNER.from_pretrained(config.model_dir)
    tokenizer = BertTokenizer.from_pretrained("/Users/tal/Desktop/Document/torch_roberta_wwm/", do_lower_case=True)
    index = 1
    opinion = []
    for line in read_data():
        ans = predict(line, model, tokenizer)
        opinion.append(ans)
    sentences = list(read_data())
    import pandas as pd
    data = pd.DataFrame({"sentence": sentences, "opinion": opinion})
    data.to_csv("pretrain_model_three_label.csv")

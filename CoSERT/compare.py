from data_augmentation import BertModel as MyBertModel
from transformers.models.bert import BertModel, BertTokenizer


def compare_output():
    tokenizer = BertTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext")
    mb = MyBertModel.from_pretrained("hfl/chinese-roberta-wwm-ext")
    bert = BertModel.from_pretrained("hfl/chinese-roberta-wwm-ext")
    sentence = "NLP数据增强的方法"
    inputs = tokenizer.encode_plus(sentence, return_tensors="pt")
    print(inputs)
    mbo = mb(**inputs)
    bo = bert(**inputs)
    print("mbo:parameters")
    for k, v in mbo.__dict__.items():
        print(k)
    print("bo:parameters")
    for k, v in bo.__dict__.items():
        print(k)

    print("*" * 20)
    print(list(mb.get_input_embeddings().parameters()))
    print(mb.get_most_recent_embedding_output())


if __name__ == '__main__':
    compare_output()

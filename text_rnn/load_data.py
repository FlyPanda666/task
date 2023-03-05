import os

from torchtext import data
from torchtext.vocab import Vectors

from utils.common.common_function import logger, get_stop_words, tokenizer, get_root_dir


def load_data(args):
    logger().info('加载数据中...')
    stop_words = get_stop_words()
    """ 如果需要设置文本的长度，则设置fix_length,否则torchtext自动将文本长度处理为最大样本长度
    text = data.Field(sequential=True, tokenize=tokenizer, fix_length=args.max_len, stop_words=stop_words)
    """
    text = data.Field(sequential=True, lower=True, tokenize=tokenizer, stop_words=stop_words)
    label = data.Field(sequential=False)

    train, val = data.TabularDataset.splits(
        path=os.path.join(get_root_dir(), "data", "textrnn"),
        skip_header=True,
        train='train.tsv',
        validation='validation.tsv',
        format='tsv',
        fields=[('index', None), ('label', label), ('text', text)],
    )

    if args.static:
        text.build_vocab(train, val, vectors=Vectors(name="data/eco_article.vector"))  # 此处为预训练的词向量
        args.embedding_dim = text.vocab.vectors.size()[-1]
        args.vectors = text.vocab.vectors

    else:
        text.build_vocab(train, val)

    label.build_vocab(train, val)

    train_iter, val_iter = data.Iterator.splits(
        (train, val),
        sort_key=lambda x: len(x.text),
        batch_sizes=(args.batch_size, len(val)),  # 训练集设置batch_size,验证集整个集合用于测试
        device=-1
    )
    args.vocab_size = len(text.vocab)
    args.label_num = len(label.vocab)
    return train_iter, val_iter, text, label

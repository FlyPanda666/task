import torchtext.data as data

from utils.common.common_function import tokenizer, get_stop_words, get_root_dir
from utils.common.my_logger import logger
import os


def load_data(args):
    logger().info("start loading data ...")
    stop_words = get_stop_words()
    text_field = data.Field(lower=True, tokenize=tokenizer, stop_words=stop_words)
    label_field = data.Field(sequential=False)

    fields = [('text', text_field), ('label', label_field)]

    train_dataset, test_dataset = data.TabularDataset.splits(
        path=os.path.join(get_root_dir(), "data_dir", "text_cnn"),
        format='tsv',
        skip_header=False,
        train='train.tsv',
        test='test.tsv',
        fields=fields
    )

    text_field.build_vocab(train_dataset, test_dataset, min_freq=5, max_size=50000)
    label_field.build_vocab(train_dataset, test_dataset)

    train_iter, test_iter = data.Iterator.splits((train_dataset, test_dataset),
                                                 batch_sizes=(args.batch_size, args.batch_size),
                                                 sort_key=lambda x: len(x.text))
    args.vocabulary_size = len(text_field.vocab)
    args.class_num = len(label_field.vocab)
    return train_iter, test_iter, text_field, label_field

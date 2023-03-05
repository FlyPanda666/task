import logging
import argparse

from utils.common import init_logger

init_logger()
logger = logging.getLogger(__name__)


def format_parser():
    parser = argparse.ArgumentParser(description='named entity recognition')
    parser.add_argument('--data_dir', type=str, default="../dataset/people_daily",
                        help='the directory where putting the original data.')
    parser.add_argument('--file_name', type=str, default="people_daily_ner_1000.txt",
                        help='the original file name.')
    parser.add_argument('--task', type=str, default="NER",
                        help='the task name you want to run.')
    parser.add_argument('--model_name_or_path', type=str,
                        default="../pretrained_models/bert-base-chinese",
                        help='the pretrained model name.')
    parser.add_argument('--max_seq_len', type=int, default=256,
                        help='the max sequence length we will focus on.')

    parser.add_argument("--train_file_name", type=str, default="train.txt", help="")
    parser.add_argument("--test_file_name", type=str, default="test.txt", help="")
    parser.add_argument("--dev_file_name", type=str, default="dev.txt", help="")
    args = parser.parse_args()
    return args

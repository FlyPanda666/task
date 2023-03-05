import logging
from people_daily_processor import load_and_cache_examples, generate_train_test_split
from arguments import format_parser
from utils.common import init_logger
from transformers import BertTokenizerFast, BertConfig
init_logger()
logger = logging.getLogger(__name__)


if __name__ == '__main__':
    args = format_parser()
    # logger.info(args.model_name_or_path)
    config = BertConfig.from_pretrained(args.model_name_or_path)
    tokenizer = BertTokenizerFast.from_pretrained(args.model_name_or_path)
    ans = load_and_cache_examples(args, tokenizer, "train")
    # logger.info(ans)
    # ans = generate_train_test_split(args)
    # logger.info(ans[:10])

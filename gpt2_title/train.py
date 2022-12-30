import argparse
import logging
import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm, trange
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import BertTokenizer
from transformers.models.gpt2.configuration_gpt2 import GPT2Config

from data_set import GPT2NewsTitleDataSet, collate_func
from model import GPT2LMHeadModel

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def train(model: GPT2LMHeadModel, device: torch.device, train_data: GPT2NewsTitleDataSet,
          test_data: GPT2NewsTitleDataSet, args: argparse):
    """模型训练函数.
    :param model:
    :param device:
    :param train_data:
    :param test_data:
    :param args:
    :return:
    """
    tb_write = SummaryWriter()
    if args.gradient_accumulation_steps < 1:
        raise ValueError("gradient_accumulation_steps参数无效，必须大于等于1")
    # 计算真实的训练batch_size大小.
    train_batch_size = int(args.train_batch_size / args.gradient_accumulation_steps)
    train_sampler = RandomSampler(train_data)
    train_data_loader = DataLoader(
        train_data, sampler=train_sampler, batch_size=train_batch_size, collate_fn=collate_func)
    total_steps = int(len(train_data_loader) * args.num_train_epochs / args.gradient_accumulation_steps)
    logger.info("总训练步数为:{}".format(total_steps))
    model.to(device)
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    # L2正则化.
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(
            nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    # 设置优化器.
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    # 动态调整学习率.
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(args.warmup_proportion * total_steps), num_training_steps=total_steps)
    # 清空cuda缓存.
    torch.cuda.empty_cache()
    # 将模型调至训练状态.
    model.train()
    title_id = train_data.title_id
    tr_loss, logging_loss, min_loss = 0.0, 0.0, 0.0
    global_step = 0
    # 开始训练模型.
    for _ in trange(0, int(args.num_train_epochs), desc="Epoch", disable=False):
        iter_bar = tqdm(train_data_loader, desc="Iter (loss=X.XXX)", disable=False)
        for step, batch in enumerate(iter_bar):
            input_ids = batch["input_ids"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            outputs = model.forward(
                input_ids=input_ids, token_type_ids=token_type_ids, labels=input_ids, title_id=title_id)
            loss = outputs[0]
            tr_loss += loss.item()
            # 将损失值放到Iter中,方便观察.
            iter_bar.set_description("Iter (loss=%5.3f)" % loss.item())
            # 判断是否进行梯度累积,如果进行,则将损失值除以累积步数.
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            # 损失进行回传.
            loss.backward()
            # 梯度剪裁.
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            # 当训练步数整除累积步数时,进行参数优化.
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                # 如果步数整除logging_steps,则记录学习率和训练集损失值.
                if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    """/Users/tal/anaconda3/lib/python3.6/site-packages/torch/optim/lr_scheduler.py:247: 
                    UserWarning: To get the last learning rate computed by the scheduler, please use `get_last_lr()`.
                    warnings.warn("To get the last learning rate computed by the scheduler, "
                    """
                    tb_write.add_scalar("lr", scheduler.get_last_lr()[0], global_step)
                    tb_write.add_scalar("train_loss", (tr_loss-logging_loss) /
                                        (args.logging_steps*args.gradient_accumulation_steps), global_step)
                    logging_loss = tr_loss
                # 如果步数整除eval_steps,则进行模型测试,记录测试集的损失.
                if args.eval_steps > 0 and global_step % args.eval_steps == 0:
                    eval_loss = evaluate(model, device, test_data, args)
                    tb_write.add_scalar("test_loss", eval_loss, global_step)
                    model.train()
        # 每个epoch进行完,保存模型.
        output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))
        model_to_save = model.module if hasattr(model, "module") else model
        model_to_save.save_pretrained(output_dir)
        # 清空cuda缓存.
        torch.cuda.empty_cache()


def evaluate(model: GPT2LMHeadModel, device: torch.device, test_data: GPT2NewsTitleDataSet, args: argparse):
    """验证模型效果.
    :param model:
    :param device:
    :param test_data:
    :param args:
    :return:
    """
    test_sampler = SequentialSampler(test_data)
    test_data_loader = DataLoader(test_data, sampler=test_sampler,
                                  batch_size=args.test_batch_size, collate_fn=collate_func)
    iter_bar = tqdm(test_data_loader, desc="iter", disable=False)
    title_id = test_data.title_id
    total_loss, total = 0.0, 0.0
    for step, batch in enumerate(iter_bar):
        # 模型设为eval
        model.eval()
        with torch.no_grad():
            input_ids = batch["input_ids"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            # 获取预测结果.
            outputs = model.forward(
                input_ids=input_ids, token_type_ids=token_type_ids, labels=input_ids, title_id=title_id)
            loss = outputs[0]
            loss = loss.item()
            # 对loss进行累加.
            total_loss += loss*len(batch["input_ids"])
            total += len(batch["input_ids"])
    # 计算最终测试集的loss结果.
    test_loss = total_loss / total
    return test_loss


def set_args():
    """设置训练模型所需参数"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='0', type=str, help='设置训练或测试时使用的显卡')
    parser.add_argument('--config_path', default='./config/config.json', type=str, help='模型参数配置信息')
    parser.add_argument('--vocab_path', default='./vocab/vocab.txt', type=str, help='词表,该词表为小词表,并增加了一些新的标记')
    parser.add_argument('--train_file_path', default='./data_dir/train_data.json', type=str, help='新闻标题生成的训练数据')
    parser.add_argument('--test_file_path', default='./data_dir/test_data.json', type=str, help='新闻标题生成的测试数据')
    parser.add_argument('--pretrained_model_path', default=None, type=str, help='预训练的GPT2模型的路径')
    parser.add_argument('--data_dir', default='./data_dir', type=str, help='生成缓存数据的存放路径')
    parser.add_argument('--num_train_epochs', default=5, type=int, help='模型训练的轮数')
    parser.add_argument('--train_batch_size', default=16, type=int, help='训练时每个batch的大小')
    parser.add_argument('--test_batch_size', default=8, type=int, help='测试时每个batch的大小')
    parser.add_argument('--learning_rate', default=1e-4, type=float, help='模型训练时的学习率')
    parser.add_argument('--warmup_proportion', default=0.1, type=float, help='warm up概率,即训练总步长的百分之多少,进行warm up')
    parser.add_argument('--adam_epsilon', default=1e-8, type=float, help='Adam优化器的epsilon值')
    parser.add_argument('--logging_steps', default=20, type=int, help='保存训练日志的步数')
    parser.add_argument('--eval_steps', default=4000, type=int, help='训练时,多少步进行一次测试')
    parser.add_argument('--gradient_accumulation_steps', default=4, type=int, help='梯度积累')
    parser.add_argument('--max_grad_norm', default=1.0, type=float, help='')
    parser.add_argument('--output_dir', default='output_dir/', type=str, help='模型输出路径')
    parser.add_argument('--seed', type=int, default=2020, help='随机种子')
    parser.add_argument('--max_len', type=int, default=512, help='输入模型的最大长度,要比config中n_ctx小')
    parser.add_argument('--title_max_len', type=int, default=32, help='生成标题的最大长度,要比max_len小')
    return parser.parse_args()


def main():
    args = set_args()
    # 设置显卡信息
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICE"] = args.device
    # 获取device信息,用于模型训练.
    device = torch.device("cuda" if torch.cuda.is_available() and int(args.device) >= 0 else "cpu")
    if args.seed:
        torch.manual_seed(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)
    # 加载模型的config.
    model_config = GPT2Config.from_json_file(args.config_path)
    if args.pretrained_model_path:
        model = GPT2LMHeadModel.from_pretrained(args.pretrained_model_path)
    else:
        # 如果没有指定的预训练模型，则初始化模型
        model = GPT2LMHeadModel(config=model_config)
    # 实例化tokenizer
    tokenizer = BertTokenizer.from_pretrained(args.vocab_path, do_lower_case=True)
    tokenizer.add_tokens("[Space]", special_tokens=True)
    # 因为是在字典中直接修改的,对于字典的大小没有变化,因此没有使用model.resize_token_embeddings(len(tokenizer))。
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    # 加载训练数据和测试数据.
    train_data = GPT2NewsTitleDataSet(
        tokenizer, args.max_len, args.title_max_len, args.data_dir, "train", args.train_file_path)
    test_data = GPT2NewsTitleDataSet(
        tokenizer, args.max_len, args.title_max_len, args.data_dir, "test", args.test_file_path)
    # 开始训练.
    train(model, device, train_data, test_data, args)


if __name__ == '__main__':
    main()

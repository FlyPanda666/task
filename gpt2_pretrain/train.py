import argparse
import json
import os
import random
from datetime import datetime

import torch
from torch.nn import DataParallel
from tqdm import tqdm, trange
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers.models.bert import BertTokenizer
from transformers.models.gpt2 import GPT2Config, GPT2LMHeadModel


def build_files(data_path: str, tokenized_data_path: str, num_pieces: int,
                full_tokenizer: BertTokenizer, min_length: int):
    """
    :param data_path: 原始文件的路径.文件的格式为json格式.
    :param tokenized_data_path: 分词之后文件保存的路径.
    :param num_pieces: 将训练语料分成多少份.
    :param full_tokenizer: 分词器.
    :param min_length: 样本的最小长度.
    :return:
    """
    with open(data_path, mode='r', encoding='utf8') as f:
        print('reading lines')
        lines = json.load(f)
        # 用[SEP]表示换行,段落之间使用SEP表示段落结束.
        lines = [line.replace('\n', ' [SEP] ') for line in lines]

    all_len = len(lines)
    if not os.path.exists(tokenized_data_path):
        os.makedirs(tokenized_data_path)
    for i in tqdm(range(num_pieces)):
        sub_lines = lines[all_len // num_pieces * i: all_len // num_pieces * (i + 1)]
        if i == num_pieces - 1:
            sub_lines.extend(lines[all_len // num_pieces * (i + 1):])
        sub_lines = [full_tokenizer.tokenize(line) for line in sub_lines if len(line) > min_length]
        sub_lines = [full_tokenizer.convert_tokens_to_ids(line) for line in sub_lines]
        full_line = []
        for sub_line in sub_lines:
            # 文章开头添加MASK表示文章开始.
            full_line.append(full_tokenizer.convert_tokens_to_ids('[MASK]'))
            full_line.extend(sub_line)
            # 文章之间添加CLS表示文章结束.
            full_line.append(full_tokenizer.convert_tokens_to_ids('[CLS]'))
        with open(tokenized_data_path + 'tokenized_train_{}.txt'.format(i), 'w') as f:
            for ids in full_line:
                f.write(str(ids) + ' ')
    print('finish')


def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='0,1,2,3', type=str, required=False, help='设置使用哪些显卡')
    parser.add_argument('--model_config', default='./models/model_config.json', type=str, required=False,
                        help='选择模型参数')
    parser.add_argument('--tokenizer_path', default='./models/vocab.txt', type=str, required=False, help='选择词库')
    parser.add_argument('--raw_data_path', default='./data_dir/试题题目和选项文本信息文本.json', type=str, required=False,
                        help='原始训练语料')
    parser.add_argument('--tokenized_data_path', default='./data_dir/tokenized_data/', type=str, required=False,
                        help='tokenized语料存放位置')
    parser.add_argument('--raw', action='store_true', help='是否先做tokenize,也就是是否从零开始构建数据集')
    parser.add_argument('--epochs', default=5, type=int, required=False, help='训练循环')
    parser.add_argument('--batch_size', default=1, type=int, required=False, help='训练batch size')
    parser.add_argument('--lr', default=1.5e-4, type=float, required=False, help='学习率')
    parser.add_argument('--warmup_steps', default=2000, type=int, required=False, help='warm up步数')
    parser.add_argument('--log_step', default=10, type=int, required=False,
                        help='多少步汇报一次loss,设置为gradient accumulation的整数倍')
    parser.add_argument('--stride', default=768, type=int, required=False, help='训练时取训练数据的窗口步长')
    parser.add_argument('--gradient_accumulation', default=1, type=int, required=False, help='梯度积累')
    parser.add_argument('--fp16', action='store_true', help='混合精度,不支持半精度的显卡请勿打开')
    parser.add_argument('--fp16_opt_level', default='O1', type=str, required=False)
    parser.add_argument('--max_grad_norm', default=1.0, type=float, required=False)
    parser.add_argument('--num_pieces', default=50, type=int, required=False, help='将训练语料分成多少份')
    parser.add_argument('--min_length', default=128, type=int, required=False, help='最短收录文章长度')
    parser.add_argument('--output_dir', default='./models/further_pretrained', type=str, required=False,
                        help='模型输出路径')
    parser.add_argument('--pretrained_model', default='uer/gpt2-chinese-cluecorpussmall', type=str, required=False,
                        help='模型训练起点路径')
    parser.add_argument('--segment', action='store_true', help='中文以词为单位')
    args = parser.parse_args()
    return args


def main():
    args = set_args()
    print('args:\n' + args.__repr__())
    # 设置程序使用哪些显卡
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    # tokenizer设置及保存
    full_tokenizer = BertTokenizer.from_pretrained("bert-base-chinese", use_fast=True)
    new_tokens = ["km/h", "m/s", "rho", "min", "LED", "kW", "dot", "times", "frac", "eta", "kWh", "mA", "Omega", "km/h",
                  "m/s"]
    full_tokenizer.add_tokens(new_tokens)
    full_tokenizer.max_len = 999999
    full_tokenizer.save_pretrained(args.output_dir)
    # config设置及保存
    model_config = GPT2Config.from_json_file(args.model_config)
    model_config.bos_token_id = full_tokenizer.mask_token_id
    model_config.eos_token_id = full_tokenizer.cls_token_id
    model_config.vocab_size = full_tokenizer.vocab_size
    model_config.save_pretrained(args.output_dir)
    print('config:\n' + model_config.to_json_string())

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('using device:', device)
    n_ctx = model_config.n_ctx
    assert args.log_step % args.gradient_accumulation == 0

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    if args.raw:
        print('building files')
        build_files(data_path=args.raw_data_path, tokenized_data_path=args.tokenized_data_path,
                    num_pieces=args.num_pieces, full_tokenizer=full_tokenizer, min_length=args.min_length)
        print('files built')

    # 模型设置
    if not args.pretrained_model:
        model = GPT2LMHeadModel(config=model_config)
    else:
        model = GPT2LMHeadModel.from_pretrained(args.pretrained_model)
    # 更新模型的tokenizer字典大小。
    model.resize_token_embeddings(len(full_tokenizer))
    model.train()
    model.to(device)
    # 统计模型参数个数。
    num_parameters = 0
    parameters = model.parameters()
    for parameter in parameters:
        num_parameters += parameter.numel()
    print('number of parameters: {}'.format(num_parameters))

    multi_gpu = False
    full_len = 0
    print('calculating total steps')
    for i in tqdm(range(args.num_pieces)):
        with open(args.tokenized_data_path + 'tokenized_train_{}.txt'.format(i), 'r') as f:
            full_len += len([int(item) for item in f.read().strip().split()])
    total_steps = int(full_len / args.stride * args.epochs / args.batch_size / args.gradient_accumulation)
    print('total steps = {}'.format(total_steps))

    optimizer = AdamW(model.parameters(), lr=args.lr, correct_bias=True)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=total_steps)

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = DataParallel(model, device_ids=[int(i) for i in args.device.split(',')])
        multi_gpu = True
    print('starting training')
    overall_step = 0
    total_loss = 0
    logging_loss = 0
    # 开始训练
    for epoch in trange(args.epochs):
        print('epoch {}'.format(epoch + 1))
        now = datetime.now()
        print('time: {}'.format(now))
        x = list(range(args.num_pieces))
        random.shuffle(x)
        for i in x:
            with open(args.tokenized_data_path + 'tokenized_train_{}.txt'.format(i), 'r') as f:
                line = f.read().strip()
            tokens = line.split()
            tokens = [int(token) for token in tokens]
            start_point = 0
            samples = []
            while start_point < len(tokens) - n_ctx:
                samples.append(tokens[start_point: start_point + n_ctx])
                start_point += args.stride
            if start_point < len(tokens):
                samples.append(tokens[len(tokens) - n_ctx:])
            random.shuffle(samples)
            for step in range(len(samples) // args.batch_size):  # drop last
                # prepare data
                batch = samples[step * args.batch_size: (step + 1) * args.batch_size]
                batch_inputs = []
                for ids in batch:
                    int_ids = [int(x) for x in ids]
                    batch_inputs.append(int_ids)
                batch_inputs = torch.tensor(batch_inputs).long().to(device)
                # forward pass
                outputs = model.forward(input_ids=batch_inputs, labels=batch_inputs)
                loss, logits = outputs[:2]
                # get loss
                if multi_gpu:
                    loss = loss.mean()
                if args.gradient_accumulation > 1:
                    loss = loss / args.gradient_accumulation
                # loss backward
                if args.fp16:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                        torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                total_loss += loss.item()
                # optimizer step
                if (overall_step + 1) % args.gradient_accumulation == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()
                if (overall_step + 1) % args.log_step == 0:
                    scale_loss = (total_loss - logging_loss) / args.log_step
                    print('Step {} epoch {}, loss {}'.format(overall_step + 1, epoch + 1, scale_loss))
                    logging_loss = total_loss
                overall_step += 1

        print('saving model for epoch {}'.format(epoch + 1))
        if not os.path.exists(os.path.join(args.output_dir, 'model_epoch{}'.format(epoch + 1))):
            os.makedirs(os.path.join(args.output_dir, 'model_epoch{}'.format(epoch + 1)))
        model_to_save = model.module if hasattr(model, 'module') else model
        model_to_save.save_pretrained(os.path.join(args.output_dir, 'model_epoch{}'.format(epoch + 1)))
        print('epoch {} finished'.format(epoch + 1))

        then = datetime.now()
        print('time: {}'.format(then))
        print('time for one epoch: {}'.format(then - now))

    print('training finished')
    model_to_save = model.module if hasattr(model, 'module') else model
    model_to_save.save_pretrained(args.output_dir)


if __name__ == '__main__':
    main()

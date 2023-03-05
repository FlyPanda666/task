import argparse
import os

import torch.cuda
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss, NLLLoss
from torch.nn import RNN

from data_load import TrainDataset, TestDataset, train_pad, test_pad
from dev import dev
from net_model import Net
from test import test
from utils.common.my_logger import logger


def train(model, iterator, optimizer, criterion):
    """

    :param model:
    :param iterator:
    :param optimizer:
    :param criterion:
    :return:
    """
    model.train()
    for i, batch in enumerate(iterator):
        tokens_x_2d, sample_id, triggers_y_2d, arguments_2d, sequence_length_1d, head_indexes_2d, mask, words_2d, triggers_2d = batch
        optimizer.zero_grad()
        trigger_logits, trigger_hat_2d, argument_logits, arguments_y_1d, argument_hat_2d = model.predict_triggers(
            tokens_x_2d, mask, head_indexes_2d, arguments_2d=None
        )
        triggers_y_2d = torch.LongTensor(triggers_y_2d).to(model.device)
        triggers_y_2d = triggers_y_2d.view(-1)
        trigger_logits = trigger_logits.view(-1, trigger_logits.shape[-1])
        trigger_loss = criterion(trigger_logits, triggers_y_2d)

        if len(argument_logits) != 1:
            argument_logits = argument_logits.view(-1, argument_logits.shape[-1])
            argument_loss = criterion(argument_logits, arguments_y_1d.view(-1))
            loss = trigger_loss + 2 * argument_loss
        else:
            loss = trigger_loss

        nn.utils.clip_grad_norm(model.parameters(), 1.0)
        loss.backward()
        optimizer.step()
        if i % 40 == 0:
            logger().info("step : {}, loss : {}".format(i, loss.item()))
        torch.cuda.empty_cache()


def parser():
    args = argparse.ArgumentParser("模型训练")
    args.add_argument("--batch_size", type=int, default=12, help="batch size 的大小")
    args.add_argument("--lr", type=float, default=0.00002, help="学习率的大小")
    args.add_argument("--log_dir", type=str, default="output", help="模型训练的日志路径")
    args.add_argument("--n_epochs", type=int, default=100, help="模型训练的轮数")
    args.add_argument("--train_set", type=str, default="", help="模型训练集路径")
    args.add_argument("--test_set", type=str, default="", help="模型测试集路径")
    args.add_argument("--dev_set", type=str, default="", help="模型验证集路径")
    args.add_argument("--sequence_length", type=int, default=50, help="文本的最大长度")

    args = args.parse_args()
    return args


if __name__ == '__main__':
    arg = parser()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Net()
    if device == "cuda":
        model.cuda()
    train_data_set = TrainDataset(arg.train_set, arg.sequence_length)
    dev_data_set = TrainDataset(arg.dev_set, arg.sequence_length)
    test_data_set = TestDataset(arg.test_set, arg.sequence_length)

    train_iter = DataLoader(
        dataset=train_data_set, batch_size=arg.batch_size, shuffle=True, num_workers=4, collate_fn=train_pad
    )
    dev_iter = DataLoader(
        dataset=dev_data_set, batch_size=arg.batch_size, shuffle=True, num_workers=4, collate_fn=train_pad
    )
    test_iter = DataLoader(
        dataset=test_data_set, batch_size=arg.batch_size, shuffle=True, num_workers=4, collate_fn=test_pad
    )

    optimizer = optim.Adam(model.parameters(), lr=arg.lr)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    # os.mkdir()创建路径中的最后一级目录，即：只创建path_03目录，而如果之前的目录不存在并且也需要创建的话，就会报错。
    # os.makedirs()创建多层目录，即：Test,path_01,path_02,path_03如果都不存在的话，会自动创建.
    # 但是如果path_03也就是最后一级目录已存在的话就会抛出FileExistsError异常。

    if not os.path.exists(arg.log_dir):
        os.makedirs(arg.log_dir)
    if not os.path.exists("models"):
        os.makedirs("models")

    early_stop = 15
    stop = 0
    best_scores = 0.0

    for epoch in range(1, arg.n_epochs + 1):
        stop += 1
        logger().info("=====train at epoch={}=====".format(epoch))
        train(model, train_iter, optimizer, criterion)
        file_name = os.path.join(arg.log_dir, str(epoch))
        logger().info("=====dev at epoch={}=====".format(epoch))
        trigger_f1, argument_f1 = dev(model, dev_iter, file_name+"_dev")
        logger().info("=====test at epoch={}=====".format(epoch))
        test(model, test_iter, file_name + "_test")

        if stop > early_stop:
            # 在得分最高之后运行15步停止。
            logger().info("the best result in epoch={}".format(epoch - early_stop - 1))
            break

        if trigger_f1 + argument_f1 > best_scores:
            best_scores = trigger_f1 + argument_f1
            stop = 0
            logger().info("the new best in epoch={}".format(epoch))
            torch.save(model, "models/best_model_filtered.pt")

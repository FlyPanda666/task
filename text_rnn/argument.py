import argparse
import torch
import os
import sys
import torch.autograd as autograd
import torch.nn.functional as F
from utils.common.common_function import logger, get_root_dir


def parse_arguments():
    """ 定义项目的参数。
    :return:
    """
    parser = argparse.ArgumentParser(description='TextRNN text classifier')
    parser.add_argument('-lr', type=float, default=0.01, help='学习率')
    parser.add_argument('-batch-size', type=int, default=128)
    parser.add_argument('-epochs', type=int, default=20)
    parser.add_argument('-embedding-dim', type=int, default=128, help='词向量的维度')
    parser.add_argument('-hidden_size', type=int, default=64, help='lstm中神经单元数')
    parser.add_argument('-layer-num', type=int, default=1, help='lstm stack的层数')
    parser.add_argument('-label-num', type=int, default=2, help='标签个数')
    parser.add_argument('-bidirectional', type=bool, default=False, help='是否使用双向lstm')
    parser.add_argument('-static', type=bool, default=False, help='是否使用预训练词向量')
    parser.add_argument('-fine-tune', type=bool, default=True, help='预训练词向量是否要微调')
    parser.add_argument('-cuda', type=bool, default=False)
    parser.add_argument('-log-interval', type=int, default=1, help='经过多少iteration记录一次训练状态')
    parser.add_argument('-test-interval', type=int, default=100, help='经过多少iteration对验证集进行测试')
    parser.add_argument('-early-stopping', type=int, default=1000, help='早停时迭代的次数')
    parser.add_argument('-save-best', type=bool, default=True, help='当得到更好的准确度是否要保存')
    parser.add_argument('-save-dir', type=str, default=os.path.join(get_root_dir(), "data", "model_dir", "textrnn"),
                        help='存储训练模型位置')

    parser.add_argument('-train', default=False, help='train a new model')
    parser.add_argument('-test', action='store_true', default=False,
                        help='test on test set, combined with -snapshot to load model')
    parser.add_argument('-predict', default=True, help='predict label of console input')

    return parser.parse_args()


def train(train_iter, dev_iter, model, config):
    """ 训练模型函数。
    :param train_iter: 训练数据集
    :param dev_iter: 验证数据集
    :param model: 模型
    :param config: 训练模型的配置参数
    :return:
    """
    logger().info("start training ...")
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    steps = 0
    best_acc = 0
    last_step = 0

    for epoch in range(1, config.epochs + 1):
        for batch in train_iter:
            feature, target = batch.text, batch.label
            feature.t_(), target.sub_(1)  # batch first, index align
            if config.cuda:
                feature, target = feature.cuda(), target.cuda()

            optimizer.zero_grad()
            logit = model(feature)

            loss = F.cross_entropy(logit, target)
            loss.backward()
            optimizer.step()

            steps += 1

            if steps % config.log_interval == 0:
                corrects = (torch.max(logit, 1)[1].view(target.size()).data == target.data).sum()
                accuracy = 100.0 * float(corrects) / batch.batch_size
                sys.stdout.write('\rBatch[{}] - loss: {:.6f}  acc: {:.3f}%({}/{})\n'.format(
                    steps, loss.data, accuracy, corrects, batch.batch_size
                ))

            if steps % config.test_interval == 0:
                dev_acc = validation(dev_iter, model, config)
                if dev_acc > best_acc:
                    best_acc = dev_acc
                    last_step = steps
                    if config.save_best:
                        logger().info("saving best model, acc : {:.4f}%\n".format(best_acc))
                        save(model, config.save_dir, 'best', steps)
                else:
                    if steps - last_step >= config.early_stopping:
                        logger().info("\nearly stop by {} steps, acc : {:.4f}%\n".format(
                            config.early_stopping, best_acc
                        ))
                        raise KeyboardInterrupt


def validation(data_iter, model, config):
    """ 验证模型函数。
    :param data_iter: 数据集
    :param model: 模型
    :param config: 配置参数
    :return:
    """
    model.eval()
    corrects, avg_loss = 0, 0
    for batch in data_iter:
        feature, target = batch.text, batch.label
        feature.t_(), target.sub_(1)
        if config.cuda:
            feature, target = feature.cuda(), target.cuda()

        logit = model(feature)
        loss = F.cross_entropy(logit, target, size_average=False)

        avg_loss += loss.data
        corrects += (torch.max(logit, 1)
                     [1].view(target.size()).data == target.data).sum()

    size = len(data_iter.dataset)
    avg_loss /= size
    accuracy = 100.0 * float(corrects) / size
    logger().info('Evaluation - loss: {:.6f}  acc: {:.3f}% ({}/{}) \n'.format(avg_loss, accuracy, corrects, size))
    return accuracy


def predict(text, model, text_field, label_field, cuda_flag):
    """ 模型预测函数
    :param text: 待预测的文本信息
    :param model: 模型
    :param text_field: 有关数据格式的字段信息
    :param label_field: 有关数据格式的字段信息
    :param cuda_flag: 是否使用GPU加速
    :return:
    """
    assert isinstance(text, str)
    model.eval()

    text = text_field.preprocess(text)
    text = [[text_field.vocab.stoi[x] for x in text]]
    x = torch.tensor(text)
    x = autograd.Variable(x)
    if cuda_flag:
        x = x.cuda()

    output = model(x)
    _, predicted = torch.max(output, 1)
    return label_field.vocab.itos[predicted.data + 1]


def save(model, save_dir, save_prefix, steps):
    """ 模型保存函数。
    :param model: 模型
    :param save_dir: 保存文件的路径
    :param save_prefix: 保存文件文件名的格式
    :param steps: 模型保存的频率
    :return:
    """
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, '{}_steps_{}.pt'.format(save_prefix, steps))
    torch.save(model.state_dict(), save_path)

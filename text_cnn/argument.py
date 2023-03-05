import os
import sys
import torch
import torch.autograd as autograd
import torch.nn.functional as F
import argparse


def parse_arguments():
    parser = argparse.ArgumentParser(description='CNN text classifier')
    # learning
    parser.add_argument('-lr', type=float, default=0.001, help='initial learning rate [default: 0.001]')
    parser.add_argument('-epochs', type=int, default=10, help='number of epochs for train [default: 10]')
    parser.add_argument('-batch-size', type=int, default=128, help='batch size for training [default: 128]')
    # wait steps
    parser.add_argument('-log-interval', type=int, default=50,
                        help='how many steps to wait before logging training status [default: 100]')
    parser.add_argument('-test-interval', type=int, default=50,
                        help='how many steps to wait before testing [default: 200]')
    parser.add_argument('-save-interval', type=int, default=50,
                        help='how many steps to wait before saving [default: 1000]')
    parser.add_argument('-save-dir', type=str, default='snapshot', help='directory to save the snapshot')
    parser.add_argument('-cuda', type=bool, default=False, help='directory to save the snapshot')
    # model
    parser.add_argument('-dropout', type=float, default=0.5, help='dropout probability [default: 0.5]')
    parser.add_argument('-embedding_dim', type=int, default=128, help='number of embedding dimension [default: 128]')
    parser.add_argument('-kernel_number', type=int, default=10, help='number of kernels')
    parser.add_argument('-vocabulary_size', type=int, default=1000, help='number of kernels')
    parser.add_argument('-channel_num', type=int, default=1, help='number of kernels')
    parser.add_argument('-class_num', type=int, default=1, help='number of kernels')
    parser.add_argument('-kernel_size', type=str, default='3,4,5',
                        help='comma-separated kernel size to use for convolution')
    # option
    parser.add_argument('-snapshot', type=str, default=None, help='filename of model snapshot [default: None]')
    parser.add_argument('-train', default=True, help='train a new model')
    parser.add_argument('-test', action='store_true', default=False,
                        help='test on test set, combined with -snapshot to load model')
    parser.add_argument('-predict', action='store_true', default=False, help='predict label of console input')
    _args = parser.parse_args()

    return _args


def train(train_iter, dev_iter, model, config):
    """ 训练模型函数。
    :param train_iter: 训练数据集
    :param dev_iter: 验证数据集
    :param model: 模型
    :param config: 训练模型的配置参数
    :return:
    """
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
                sys.stdout.write('\rBatch[{}] - loss: {:.6f}  acc: {:.3f}%({}/{})'.format(
                    steps, loss.data, accuracy, corrects, batch.batch_size
                ))
            if steps % config.test_interval == 0:
                dev_acc = test(dev_iter, model, config)
                if dev_acc > best_acc:
                    best_acc = dev_acc
                    last_step = steps
                    save(model, config.save_dir, 'best', steps)
            if steps % config.save_interval == 0:
                save(model, config.save_dir, 'snapshot', steps)


def test(data_iter, model, config):
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
    print('Evaluation - loss: {:.6f}  acc: {:.3f}% ({}/{}) \n'.format(avg_loss, accuracy, corrects, size))
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

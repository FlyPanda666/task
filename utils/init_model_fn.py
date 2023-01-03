import torch.nn as nn
from transformers.models.roberta import RobertaConfig, RobertaModel


def init_network(model, method='xavier', exclude='embedding', seed=123):
    """对网络进行初始化.
    :param model: 需要初始化的模型.
    :param method: 初始化采用的方法.
    :param exclude: 不需要初始化的层中包括的关键字.
    :param seed: 随机种子数.
    :return:
    """
    for name, w in model.named_parameters():
        if exclude not in name:
            if len(w.size()) < 2:
                continue
            if 'weight' in name:
                if method == 'xavier':
                    nn.init.xavier_normal_(w)
                elif method == 'kaiming':
                    nn.init.kaiming_normal_(w)
                else:
                    nn.init.normal_(w)
            elif 'bias' in name:
                nn.init.constant_(w, 0)
            else:
                pass


if __name__ == '__main__':
    net = RobertaModel(RobertaConfig())
    for k, v in net.named_parameters():
        if len(v.shape) >= 2 and "weight" in k and "embedding" not in k:
            print(k, v)
            break

    init_network(model=net)
    for k, v in net.named_parameters():
        if len(v.shape) >= 2 and "weight" in k and "embedding" not in k:
            print(k, v)
            break

    init_network(model=net, seed=4027)
    for k, v in net.named_parameters():
        if len(v.shape) >= 2 and "weight" in k and "embedding" not in k:
            print(k, v)
            break

    init_network(model=net, method="kaiming")
    for k, v in net.named_parameters():
        if len(v.shape) >= 2 and "weight" in k and "embedding" not in k:
            print(k, v)
            break

from transformers.models.bert import BertModel, BertConfig
import torch
import random


class GradientStorage:
    """此类对象存储给定 PyTorch 模块输出的中间梯度，否则可能不会保留.
    """
    def __init__(self, module):
        self._stored_gradient = None
        module.register_backward_hook(self.hook)

    def hook(self, module, grad_in, grad_out):
        self._stored_gradient = grad_out[0]

    def get(self):
        return self._stored_gradient


if __name__ == '__main__':
    # net = BertModel.from_pretrained("bert-base-chinese")
    # con = BertConfig.from_pretrained("bert-base-chinese")
    # for k, v in net.named_parameters():
    #     if "word_embeddings" in k:
    #         print(v)
    # word_embeddings = net.embeddings.word_embeddings
    # gs = GradientStorage(word_embeddings)
    # grad = gs.get()
    #
    x = torch.randn([3, 4])
    # print(x)
    # # 将x中的每一个元素与0.5进行比较
    # # 当元素大于等于0.5返回True,否则返回False
    # mask = x.ge(0.5)
    # print(mask)
    # print(torch.masked_select(x, mask))
    # print(x.masked_select(mask))
    #
    # print(x.sum(dim=0))
    print(random.randrange(10))
    print(torch.zeros(5))
    print(x)
    print(x.topk(1))

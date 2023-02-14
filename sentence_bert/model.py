import torch
from torch import nn
from transformers.models.bert import BertConfig, BertModel


class SentenceBert(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = BertConfig.from_pretrained("bert-base-chinese")
        self.model = BertModel.from_pretrained("bert-base-chinese")

    def forward(self, s1_input_ids, s2_input_ids):
        s1_mask = torch.ne(s1_input_ids, 0)
        s2_mask = torch.ne(s2_input_ids, 0)

        s1_output = self.model(input_ids=s1_input_ids, attention_mask=s1_mask)
        s2_output = self.model(input_ids=s2_input_ids, attention_mask=s2_mask)

        s1_vec = s1_output[1]
        s2_vec = s2_output[1]
        return s1_vec, s2_vec


if __name__ == '__main__':
    a = torch.tensor([1, 1, 1, 1, 0, 0])
    mask = torch.ne(a, 0).long()
    print(mask)

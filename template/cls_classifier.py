import torch
from torch import nn
from transformers.models.bert import BertConfig, BertModel


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.config = BertConfig.from_pretrained("bert-base-chinese")
        self.bert = BertModel.from_pretrained("bert-base-chinese")

    def forward(self, input_ids, attention_mask, encoder_type='fist-last-avg'):
        """
        :param input_ids:
        :param attention_mask:
        :param encoder_type: encoder_type:  "first-last-avg", "last-avg", "cls", "pooler(cls + dense)"
        :return:
        """
        output = self.bert(input_ids, attention_mask, output_hidden_states=True)

        if encoder_type == 'fist-last-avg':
            # 取出第一层和最后一层的隐状态,然后经过平均池化
            first = output.hidden_states[1]
            last = output.hidden_states[-1]
            seq_length = first.size(1)

            first_avg = torch.avg_pool1d(first.transpose(1, 2), kernel_size=seq_length).squeeze(-1)  # batch, hid_size
            last_avg = torch.avg_pool1d(last.transpose(1, 2), kernel_size=seq_length).squeeze(-1)  # batch, hid_size
            final_encoding = torch.avg_pool1d(
                torch.cat([first_avg.unsqueeze(1), last_avg.unsqueeze(1)], dim=1).transpose(1, 2), kernel_size=2).\
                squeeze(-1)
            return final_encoding

        if encoder_type == 'last-avg':
            sequence_output = output.last_hidden_state  # (batch_size, max_len, hidden_size)
            seq_length = sequence_output.size(1)
            final_encoding = torch.avg_pool1d(sequence_output.transpose(1, 2), kernel_size=seq_length).squeeze(-1)
            return final_encoding

        if encoder_type == "cls":
            sequence_output = output.last_hidden_state
            cls = sequence_output[:, 0]  # [b,d]
            return cls

        if encoder_type == "pooler":
            pooler_output = output.pooler_output  # [b,d]
            return pooler_output

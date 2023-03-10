from transformers.models.bert import BertModel, BertPreTrainedModel
from torch.nn.utils.rnn import pad_sequence
from torchcrf import CRF
import torch.nn as nn


class BertNER(BertPreTrainedModel):
    def __init__(self, config):
        super(BertNER, self).__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.crf = CRF(config.num_labels, batch_first=True)

        self.init_weights()

    def forward(self,
                input_data,
                token_type_ids=None,
                attention_mask=None,
                labels=None,
                position_ids=None,
                inputs_embeds=None,
                head_mask=None):
        input_ids, input_token_starts = input_data
        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask,
                            inputs_embeds=inputs_embeds)
        # (batch_size, sequence_length, hidden_size)
        sequence_output = outputs[0]

        # 去除[CLS]标签等位置，获得与label对齐的pre_label表示,根据start_index这个矩阵，拿到输出中我们关注的信息，也就是非CLS和padding字符的输出向量。
        origin_sequence_output = [
            layer[starts.nonzero().squeeze(1)]
            for layer, starts in zip(sequence_output, input_token_starts)
        ]
        # 将sequence_output的pred_label维度padding到最大长度
        # batch_size * seq_len * hidden_size
        padded_sequence_output = pad_sequence(origin_sequence_output, batch_first=True)
        # dropout pred_label的一部分feature
        padded_sequence_output = self.dropout(padded_sequence_output)
        # 得到判别值 batch_size * seq_len * num_labels
        logits = self.classifier(padded_sequence_output)
        outputs = (logits, )
        if labels is not None:
            # 在pytorch中这种mask本质上是bool值。
            loss_mask = labels.gt(-1)
            loss = self.crf(logits, labels, loss_mask) * (-1)
            outputs = (loss, ) + outputs

        # contain: (loss), scores
        return outputs


if __name__ == "__main__":
    pass

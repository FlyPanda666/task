import torch.nn as nn
from transformers import BertPreTrainedModel, BertModel
from transformers.models.bert import BertModel


class NerBert(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.label_num = config.label_num
        self.bert = BertModel(config)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.label_num)
        self.post_init()

    def forward(self, all_input_ids=None, all_attention_mask=None, all_token_type_ids=None, all_label_ids=None):
        output = self.bert(input_ids=all_input_ids,
                           attention_mask=all_attention_mask,
                           token_type_ids=all_token_type_ids)
        token_representation = output[0]
        output_classifier = self.classifier(token_representation)
        return output_classifier

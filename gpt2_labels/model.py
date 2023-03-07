from torch.nn import CrossEntropyLoss
import torch.nn as nn
from transformers.models.gpt2 import GPT2PreTrainedModel, GPT2Model


class GPT2LMHeadModel(GPT2PreTrainedModel):

    def _reorder_cache(self, past, beam_idx):
        pass

    def __init__(self, config):
        super().__init__(config)
        self.transformer = GPT2Model(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.init_weights()

    def forward(self, input_ids=None, token_type_ids=None, labels=None, label_id=None):
        """
        :param input_ids: 输入序列在词表中的索引序列,size:[batch_size, sequence_length]
        :param token_type_ids: 用于区分输入序列中content和label的分隔符序列,size:[batch_size, sequence_length].
        :param labels: 标签序列,size:[batch_size, sequence_length],一般情况下,与input_ids相同.
        :param label_id: label部分分隔符的id.
        :return:
        """
        # (last_hidden_state, presents, all_hidden_states, attentions, cross_attentions)
        transformer_outputs = self.transformer(input_ids=input_ids, token_type_ids=token_type_ids)
        hidden_states = transformer_outputs[0]
        lm_logits = self.lm_head(hidden_states)
        outputs = (lm_logits,) + transformer_outputs[1:]
        if labels is not None:
            if label_id is None or token_type_ids is None:
                raise Exception("当labels不为None时,title_id和token_type_ids均不可以为None!")
            mask = (token_type_ids == label_id).long()
            labels = labels * mask
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = CrossEntropyLoss(ignore_index=0, reduction="sum")
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            num = shift_labels.ne(0).long().sum().item()
            loss = loss / num
            outputs = (loss,) + outputs
        # (loss), lm_logits, presents, (all hidden_states), (attentions)
        return outputs

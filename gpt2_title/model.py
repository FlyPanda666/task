from torch.nn import CrossEntropyLoss
import torch.nn as nn
from transformers.models.gpt2 import GPT2PreTrainedModel, GPT2Model


class GPT2LMHeadModel(GPT2PreTrainedModel):
    """GPT2模型"""

    def _reorder_cache(self, past, beam_idx):
        pass

    def __init__(self, config):
        super().__init__(config)
        self.transformer = GPT2Model(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.init_weights()

    def forward(self, input_ids=None, token_type_ids=None, labels=None, title_id=None):
        """
        :param input_ids: 输入序列在词表中的索引序列,size:[batch_size, sequence_length]
        :param token_type_ids: 用于区分输入序列中content和title的分隔符序列,size:[batch_size, sequence_length].
        :param labels: 标签序列,size:[batch_size, sequence_length],一般情况下,与input_ids相同.
        :param title_id: title部分分隔符的id.
        :return:
        """
        transformer_outputs = self.transformer(input_ids, token_type_ids=token_type_ids)
        # 获取GPT2模型的最后一层的隐层节点状态，size:[batch_size, sequence_length, config.n_embed]
        hidden_states = transformer_outputs[0]
        lm_logits = self.lm_head(hidden_states)
        outputs = (lm_logits,) + transformer_outputs[1:]
        if labels is not None:
            # 计算loss时,title_id不可以为None,因为需要title_id找到title的部分
            if title_id is None or token_type_ids is None:
                raise Exception("当labels不为None时,title_id和token_type_ids均不可以为None!")
            # 获取mask值,如果token_type_ids中等于title_id的部分需要计算loss,标记为1;否则为0.
            # size:[batch_size, sequence_length]
            mask = (token_type_ids == title_id).long()
            labels = labels * mask
            # input_ids中的第一个token的预测结果,实际上是标签中的第二个token,以此类推,最终仅计算sequence_length-1个token的loss.
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            # 定义损失函数CrossEntropyLoss,并且设置忽略计算loss的索引,以及返回loss的形式,忽略shift_labels中为0的loss，也就是仅计算title部分的损失值
            # 对loss的计算方式为sum，由于我们仅计算了title部分的损失值,如果使用mean,会使loss变小(实际除的是sequence_length-1,不是title部分的真实长度).
            loss_fct = CrossEntropyLoss(ignore_index=0, reduction="sum")
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            # 获取title部分的真实长度,并计算真实loss.
            num = shift_labels.ne(0).long().sum().item()
            loss = loss / num
            outputs = (loss,) + outputs
        # (loss), lm_logits, presents, (all hidden_states), (attentions)
        return outputs

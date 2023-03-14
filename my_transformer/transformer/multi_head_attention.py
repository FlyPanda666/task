import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0., bias=True):
        """
        :param embed_dim: 词嵌入的维度,也就是前面的d_model参数,论文中的默认值为512
        :param num_heads: 多头注意力机制中多头的数量,也就是前面的nhead参数,论文默认值为 8
        :param dropout:
        :param bias: 最后对多头的注意力（组合）输出进行线性变换时,是否使用偏置.
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        assert self.embed_dim % num_heads == 0, "embed_dim 除以 num_heads必须为整数"
        self.head_dim = embed_dim // num_heads
        self.dropout = dropout

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def forward(self, query, key, value, attn_mask=None, key_padding_mask=None):
        """编码时query,key,value都是同一个输入,解码时的输入也都是同一个输入,解码和编码交互时key,value指的是memory,query指的是tgt.
        :param query: [tgt_len, batch_size, embed_dim],tgt_len 表示目标序列的长度.
        :param key: [src_len, batch_size, embed_dim],src_len 表示源序列的长度.
        :param value: [src_len, batch_size, embed_dim],src_len 表示源序列的长度.
        :param attn_mask: [tgt_len, src_len] or [num_heads*batch_size,tgt_len, src_len]
        一般只在解码时使用,为了并行一次喂入所有解码部分的输入,所以要用mask来进行掩盖当前时刻之后的位置信息.
        :param key_padding_mask: [batch_size, src_len], src_len 表示源序列的长度.
        :return:
        attn_output: [tgt_len, batch_size, embed_dim]
        attn_output_weights: [batch_size, tgt_len, src_len]
        """
        return multi_head_attention_forward(query, key, value, self.num_heads,
                                            self.dropout,
                                            out_proj=self.out_proj,
                                            training=self.training,
                                            key_padding_mask=key_padding_mask,
                                            q_proj=self.q_proj,
                                            k_proj=self.k_proj,
                                            v_proj=self.v_proj,
                                            attn_mask=attn_mask)


def multi_head_attention_forward(
        query, key, value, num_heads, dropout_p, out_proj, training=True, key_padding_mask=None, q_proj=None,
        k_proj=None, v_proj=None, attn_mask=None,
):
    q = q_proj(query)
    k = k_proj(key)
    v = v_proj(value)
    tgt_len, bsz, embed_dim = query.size()
    src_len = key.size(0)
    head_dim = embed_dim // num_heads
    scaling = float(head_dim) ** -0.5
    q = q * scaling

    if attn_mask is not None:
        if attn_mask.dim() == 2:
            attn_mask = attn_mask.unsqueeze(0)  # [1, tgt_len, src_len]
            if list(attn_mask.size()) != [1, query.size(0), key.size(0)]:
                raise RuntimeError('The size of the 2D attn_mask is not correct.')
        elif attn_mask.dim() == 3:
            if list(attn_mask.size()) != [bsz * num_heads, query.size(0), key.size(0)]:
                raise RuntimeError('The size of the 3D attn_mask is not correct.')

    q = q.contiguous().view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
    k = k.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
    v = v.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
    attn_output_weights = torch.bmm(q, k.transpose(1, 2))
    # attn_output_weights: shape,[batch_size * num_heads, tgt_len, src_len] 这就num_heads个QK相乘后的注意力矩阵

    if attn_mask is not None:
        attn_output_weights += attn_mask

    if key_padding_mask is not None:
        attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
        # 扩展维度，从[batch_size,src_len]变成[batch_size,1,1,src_len]
        attn_output_weights = attn_output_weights.masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(2), float('-inf'))
        attn_output_weights = attn_output_weights.view(bsz * num_heads, tgt_len, src_len)

    # [batch_size * num_heads, tgt_len, src_len]
    attn_output_weights = F.softmax(attn_output_weights, dim=-1)
    attn_output_weights = F.dropout(attn_output_weights, p=dropout_p, training=training)
    attn_output = torch.bmm(attn_output_weights, v)
    # [batch_size * num_heads,tgt_len, head_dim]

    attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
    attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)

    out = out_proj(attn_output)
    # 这里就是多个z  线性组合成Z  [tgt_len, batch_size, embed_dim]

    # average attention weights over heads
    return out, attn_output_weights.sum(dim=1) / num_heads

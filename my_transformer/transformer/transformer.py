import copy
import torch
import torch.nn as nn
from multi_head_attention import MultiHeadAttention
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_


def _get_clones(module, num_layers):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(num_layers)])


class EncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        """
        :param d_model: d_k = d_v = d_model/nhead = 64,模型中向量的维度,论文默认值为512
        :param nhead: 多头注意力机制中多头的数量,论文默认为值 8
        :param dim_feedforward: 全连接中向量的维度,论文默认值为 2048
        :param dropout: 丢弃率,论文中的默认值为 0.1
        """
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, nhead, dropout=dropout)

        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.activation = F.relu

        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        """
        :param src: 编码部分的输入,形状为 [src_len, batch_size, embed_dim]
        :param src_mask: 编码部分的attention mask.
        :param src_key_padding_mask: 编码部分输入的padding情况,形状为 [batch_size, src_len]
        :return:
        """
        src2 = self.self_attn(src, src, src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        src2 = self.activation(self.linear1(src))
        src2 = self.linear2(self.dropout(src2))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


class Encoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, mask=None, src_key_padding_mask=None):
        """
        :param src: 编码部分的输入,形状为 [src_len, batch_size, embed_dim]
        :param mask:
        :param src_key_padding_mask: 编码部分输入的padding情况,形状为 [batch_size, src_len]
        :return:
        """
        output = src
        for module in self.layers:
            output = module(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)
        if self.norm is not None:
            output = self.norm(output)
        # [src_len, batch_size, embed_dim]
        return output


class DecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        """
        :param d_model: d_k = d_v = d_model/nhead = 64,模型中向量的维度,论文默认值为 512
        :param nhead: 多头注意力机制中多头的数量,论文默认为值 8
        :param dim_feedforward: 全连接中向量的维度,论文默认值为 2048
        :param dropout: 丢弃率,论文中的默认值为 0.1
        """
        super().__init__()

        self.self_attn = MultiHeadAttention(embed_dim=d_model, num_heads=nhead, dropout=dropout)
        # 解码部分输入序列之间的多头注意力（也就是论文结构图中的Masked Multi-head attention)
        self.multi_head_attn = MultiHeadAttention(embed_dim=d_model, num_heads=nhead, dropout=dropout)
        # 编码部分输出（memory）和解码部分之间的多头注意力机制。
        # Implementation of Feedforward model

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = F.relu

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None,
                memory_key_padding_mask=None):
        """
        :param tgt: 解码部分的输入,[tgt_len, batch_size, embed_dim]
        :param memory: 编码部分的输出,[src_len, batch_size, embed_dim]
        :param tgt_mask: 注意力Mask输入,用于掩盖当前position之后的信息,[tgt_len, tgt_len]
        :param memory_mask: 编码器-解码器交互时的注意力掩码,一般为None.
        :param tgt_key_padding_mask: 解码部分输入的padding情况,[batch_size, tgt_len]
        :param memory_key_padding_mask: 编码部分输入的padding情况,[batch_size, src_len]
        :return:
        """
        tgt2 = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)[0]
        # 解码部分输入序列之间的多头注意力（也就是论文结构图中的Masked Multi-head attention)
        tgt = tgt + self.dropout1(tgt2)  # 接着是残差连接
        tgt = self.norm1(tgt)

        tgt2 = self.multihead_attn(
            tgt, memory, memory, attn_mask=memory_mask, key_padding_mask=memory_key_padding_mask)[0]

        # 解码部分的输入经过多头注意力后同编码部分的输出（memory）通过多头注意力机制进行交互
        tgt = tgt + self.dropout2(tgt2)  # 残差连接
        tgt = self.norm2(tgt)

        tgt2 = self.activation(self.linear1(tgt))
        tgt2 = self.linear2(self.dropout(tgt2))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt


class Decoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None,
                memory_key_padding_mask=None):
        """
        :param tgt: 解码部分的输入,形状为 [tgt_len, batch_size, embed_dim]
        :param memory: 编码部分最后一层的输出 [src_len, batch_size, embed_dim]
        :param tgt_mask: 注意力Mask输入，用于掩盖当前position之后的信息, [tgt_len, tgt_len]
        :param memory_mask: 编码器-解码器交互时的注意力掩码，一般为None
        :param tgt_key_padding_mask: 解码部分输入的padding情况，形状为 [batch_size, tgt_len]
        :param memory_key_padding_mask: 编码部分输入的padding情况，形状为 [batch_size, src_len]
        :return:
        """
        output = tgt
        for module in self.layers:
            output = module(output, memory, tgt_mask=tgt_mask, memory_mask=memory_mask,
                            tgt_key_padding_mask=tgt_key_padding_mask,
                            memory_key_padding_mask=memory_key_padding_mask)
        if self.norm is not None:
            output = self.norm(output)
        # [tgt_len, batch_size, embed_dim]
        return output


class Trans(nn.Module):
    def __init__(
            self, d_model=512, nhead=8, num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=2048, dropout=0.1):
        """
        :param d_model: d_k = d_v = d_model/nhead = 64,模型中向量的维度,论文默认值为 512
        :param nhead: 多头注意力机制中多头的数量,论文默认为值 8
        :param num_encoder_layers: encoder堆叠的数量,也就是论文中的n,论文默认值为6
        :param num_decoder_layers: decoder堆叠的数量,也就是论文中的n,论文默认值为6
        :param dim_feedforward: 全连接中向量的维度,论文默认值为 2048
        :param dropout: 丢弃率,论文中的默认值为 0.1
        """
        super().__init__()
        encoder_layer = EncoderLayer(d_model, nhead, dim_feedforward, dropout)
        encoder_norm = nn.LayerNorm(d_model)
        self.encoder = Encoder(encoder_layer, num_encoder_layers, encoder_norm)

        decoder_layer = DecoderLayer(d_model, nhead, dim_feedforward, dropout)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = Decoder(decoder_layer, num_decoder_layers, decoder_norm)

        self._reset_parameters()
        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None, memory_mask=None, src_key_padding_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None):
        """
        :param src: [src_len, batch_size, embed_dim]
        :param tgt: [tgt_len, batch_size, embed_dim]
        :param src_mask: None
        :param tgt_mask: [tgt_len, tgt_len]
        :param memory_mask: None
        :param src_key_padding_mask: [batch_size, src_len]
        :param tgt_key_padding_mask: [batch_size, tgt_len]
        :param memory_key_padding_mask:  [batch_size, src_len]
        :return: [tgt_len,batch_size,embed_dim]
        """
        memory = self.encoder(src, mask=src_mask, src_key_padding_mask=src_key_padding_mask)
        # [src_len, batch_size, embed_dim]
        output = self.decoder(tgt=tgt, memory=memory, tgt_mask=tgt_mask, memory_mask=memory_mask,
                              tgt_key_padding_mask=tgt_key_padding_mask,
                              memory_key_padding_mask=memory_key_padding_mask)
        # [tgt_len, batch_size, embed_dim]
        return output

    @staticmethod
    def generate_square_subsequent_mask(sz: int):
        r"""Generate a square mask for the sequence. The masked positions are filled with float('-inf').
            Unmasked positions are filled with float(0.0).
        """
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask


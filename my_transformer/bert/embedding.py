import torch
import torch.nn as nn


class PositionalEmbedding(nn.Module):
    def __init__(self, hidden_size, max_position_embeddings=512, initializer_range=0.02):
        super().__init__()
        assert max_position_embeddings >= 512, "config.max_position_embeddings参数必须大于等于512"
        self.embedding = nn.Embedding(max_position_embeddings, hidden_size)
        self._reset_parameter(initializer_range)

    def forward(self, position_ids):
        return self.embedding(position_ids).transpose(0, 1)

    def _reset_parameter(self, initializer_range):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.normal_(p, mean=0.0, std=initializer_range)
                # nn.init.ones_(p)


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, hidden_size, pad_token_id=0, initializer_range=0.02):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=pad_token_id)
        self._reset_parameters(initializer_range)

    def forward(self, input_ids):
        return self.embedding(input_ids)

    def _reset_parameters(self, initializer_range):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.normal_(p, mean=0.0, std=initializer_range)


class SegmentEmbedding(nn.Module):
    def __init__(self, type_vocab_size, hidden_size, initializer_range=0.02):
        super(SegmentEmbedding, self).__init__()
        self.embedding = nn.Embedding(type_vocab_size, hidden_size)
        self._reset_parameters(initializer_range)

    def forward(self, token_type_ids):
        return self.embedding(token_type_ids)

    def _reset_parameters(self, initializer_range):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.normal_(p, mean=0.0, std=initializer_range)


class BertEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.word_embeddings = TokenEmbedding(
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_size,
            pad_token_id=config.pad_token_id,
            initializer_range=config.initializer_range)
        # return shape [src_len, batch_size, hidden_size]

        self.position_embeddings = PositionalEmbedding(
            max_position_embeddings=config.max_position_embeddings,
            hidden_size=config.hidden_size,
            initializer_range=config.initializer_range)
        # return shape [src_len, 1, hidden_size]

        self.token_type_embeddings = SegmentEmbedding(
            type_vocab_size=config.type_vocab_size,
            hidden_size=config.hidden_size,
            initializer_range=config.initializer_range)
        # return shape  [src_len, batch_size, hidden_size]

        self.LayerNorm = nn.LayerNorm(config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))
        # shape: [1, max_position_embeddings]

    def forward(self, input_ids=None, position_ids=None, token_type_ids=None):
        """
        :param input_ids: 输入序列的原始token id,shape: [src_len, batch_size]
        :param position_ids: 位置序列,本质就是 [0,1,2,3,...,src_len-1],shape: [1,src_len]
        :param token_type_ids: 句子分隔token,例如[0,0,0,0,1,1,1,1]用于区分两个句子,shape:[src_len,batch_size]
        :return: [src_len, batch_size, hidden_size]
        """
        src_len = input_ids.size(0)
        token_embedding = self.word_embeddings(input_ids)
        # 在实际中这个参数可以不传.
        if position_ids is None:
            position_ids = self.position_ids[:, :src_len]
        positional_embedding = self.position_embeddings(position_ids)
        # 如果输入模型的只有一个序列,那么这个参数也不用传值,如果是两个序列这个参数是必须要传的,用来区分不同的句子信息.
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids, device=self.position_ids.device)
        segment_embedding = self.token_type_embeddings(token_type_ids)
        embeddings = token_embedding + positional_embedding + segment_embedding
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

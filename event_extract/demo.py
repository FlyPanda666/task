import logging
from dataclasses import dataclass, asdict
import jsonpickle
import torch
import torch.nn as nn
from copy import deepcopy
from typing import *
from abc import abstractmethod
from utils.common import init_logger

init_logger()
logger = logging.getLogger(__name__)


@dataclass
class Config:
    encoder_hidden_dim: int = 768
    decoder_hidden_dim: int = 768
    teacher_forcing_ratio: float = 0.5
    dropout_prob: float = 0.5
    src_vocab_size: int = 1000
    tgt_vocab_size: int = 2000
    tgt_max_length: int = 10
    src_embedding_dim: int = 512
    tgt_embedding_dim: int = 512
    src_sequence_length: int = 100
    # tgt_sequence_length 因为输入序列每次输入一个token,这里不再提供这个参数.
    encoder_bidirectional: bool = True
    # decoder_bidirectional 在decoder中不能使用双向,这里不再提供这个参数.
    encoder_num_layers: int = 2
    decoder_num_layers: int = 2
    encoder_layer_type: nn.Module = nn.GRU
    decoder_layer_type: nn.Module = nn.GRU

    batch_size: int = 16

    def __repr__(self):
        return jsonpickle.dumps(asdict(self), indent=2)


class UpdateConfigMixin:

    CONFIG: Config = Config()

    def _update_self_config(self, kwargs):
        cls_config = deepcopy(self.__class__.CONFIG)
        for k, v in kwargs.items():
            setattr(cls_config, k, v)
        return cls_config

    def show(self, **kwargs):
        return self._update_self_config(kwargs)


class A(UpdateConfigMixin):
    CONFIG: Config = Config()
    pass


class Encoder(nn.Module, UpdateConfigMixin):
    CONFIG: Config = Config()

    def __init__(
        self,
        **kwargs,
    ):
        super().__init__()
        self.config = self._update_self_config(kwargs)
        self.layer = self._load_encoder_layer()
        self.embedding = nn.Embedding(
            self.config.src_vocab_size, self.config.src_embedding_dim)
        self.dropout = nn.Dropout(self.config.dropout_prob)

    def _load_encoder_layer(self):
        rnn_type = self.config.encoder_layer_type
        return rnn_type(self.config.src_embedding_dim,
                        self.config.encoder_hidden_dim)

    def forward(self, src_inputs: torch.Tensor):
        print(src_inputs.shape)
        src_input_tensor = self.embedding(src_inputs)
        print(src_input_tensor.shape)
        output, hidden_states = self.layer(src_input_tensor)
        return output, hidden_states

    # def _update_self_config(self, kwargs):
    #     cls_config = deepcopy(self.__class__.CONFIG)
    #     for k, v in kwargs.items():
    #         setattr(cls_config, k, v)
    #     return cls_config


class Decoder(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.layer = self._load_decoder_layer(config)

    def _load_decoder_layer(self, config):
        raise NotImplementedError

    def forward(self, tgt_input: torch.Tensor, context: torch.Tensor):
        raise NotADirectoryError


class Attention(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()

    @abstractmethod
    def forward(self, query: torch.Tensor, key: torch.Tensor):
        pass


class Seq2Seq(nn.Module):

    TEACHER_FORCING_RATIO: float = 0.5

    def __init__(self, encoder: Encoder, decoder: Decoder, attention: Attention):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.attention = attention
        self.teacher_forcing_ratio = self.__class__.TEACHER_FORCING_RATIO

    @abstractmethod
    def forward(self, src_inputs: torch.Tensor, tgt_input: torch.Tensor):
        pass


if __name__ == "__main__":
    batch_size = 64
    output_dir = "./temp/output_dir"
    src_sequence_length = 500
    labels_num = 22
    encoder = Encoder(batch_size=batch_size, output_dir=output_dir,
                      src_sequence_length=src_sequence_length, labels_num=labels_num)
    # print(encoder)
    # x = torch.randint(300, (Config().src_sequence_length, Config().batch_size))
    # print(x.shape)
    # out, hidden = encoder(x)
    # print(out.shape, hidden.shape)
    # print(encoder.config)
    a = A().show()
    print(a)

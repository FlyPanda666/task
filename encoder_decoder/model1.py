"""
import torch

print(torch.cuda.is_available())
!pip install torchtext==0.6.0
!python -m spacy download en_core_web_sm
!python -m spacy download de_core_news_sm
"""

import math
import random
import time

import numpy as np
import spacy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.modules.rnn import LSTM
from torchtext.data import Field, BucketIterator
from torchtext.datasets import Multi30k

SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

spacy_de = spacy.load("de_core_news_sm")
spacy_en = spacy.load("en_core_web_sm")


def tokenize_de(text):
    return [tok.text for tok in spacy_de.tokenizer(text)][::-1]


def tokenize_en(text):
    return [tok.text for tok in spacy_en.tokenizer(text)]


SRC = Field(tokenize=tokenize_de,
            init_token='<sos>',
            eos_token='<eos>',
            lower=True)

TRG = Field(tokenize=tokenize_en,
            init_token='<sos>',
            eos_token='<eos>',
            lower=True)

train_data, valid_data, test_data = Multi30k.splits(exts=('.de', '.en'), fields=(SRC, TRG))
print(f"Number of training examples: {len(train_data.examples)}")
print(f"Number of validation examples: {len(valid_data.examples)}")
print(f"Number of testing examples: {len(test_data.examples)}")

print(vars(train_data.examples[0]))

# 构建字典.
SRC.build_vocab(train_data, min_freq=2)
TRG.build_vocab(train_data, min_freq=2)
print(f"Unique tokens in source (de) vocabulary: {len(SRC.vocab)}")
print(f"Unique tokens in target (en) vocabulary: {len(TRG.vocab)}")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 128

train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
    (train_data, valid_data, test_data),
    batch_size=BATCH_SIZE,
    device=device)

print(train_iterator.data()[0])
print(valid_iterator.data()[:20])
print(test_iterator.data()[:20])


class Encoder(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, num_layers, dropout):
        """
        :param input_dim: source的字典大小.
        :param embedding_dim: 词向量的维度.
        :param hidden_dim: 隐状态的维度.
        :param num_layers: RNN的层数.
        :param dropout:
        """
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=input_dim, embedding_dim=embedding_dim)
        self.dropout = nn.Dropout(dropout)
        self.rnn = LSTM(input_size=embedding_dim, hidden_size=hidden_dim, num_layers=num_layers, dropout=dropout)

    def forward(self, src):
        """
        :param src: src batch_size
        :return:
        """
        embed = self.embedding(src)
        embed = self.dropout(embed)
        # outputs: [src, batch_size, hidden_dim*directions]
        # hidden: [num_layers*directions, batch_size, hidden_dim]
        # cell: [num_layers*directions, batch_size, hidden_dim]
        outputs, (hidden, cell) = self.rnn(embed)
        return hidden, cell


class Decoder(nn.Module):
    def __init__(self, output_dim, embedding_dim, hidden_dim, num_layers, dropout):
        """
        :param output_dim: target的字典大小.
        :param embedding_dim: 词向量的维度.
        :param hidden_dim: 隐状态的维度.
        :param num_layers: RNN的层数.
        :param dropout:
        """
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=output_dim, embedding_dim=embedding_dim)
        self.rnn = LSTM(input_size=embedding_dim, hidden_size=hidden_dim, num_layers=num_layers, dropout=dropout)
        self.fc_out = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs, hidden, cell):
        """
        :param inputs: 当前时刻的输入,只有一个token.
        :param hidden: 上下文.
        :param cell: 上下文.
        :return:
        """
        inputs = inputs.unsqueeze(0)
        inputs_embed = self.embedding(inputs)
        inputs_embed = self.dropout(inputs_embed)

        # outputs: [1, batch_size, hidden_dim*directions]
        outputs, (hidden, cell) = self.rnn(inputs_embed, (hidden, cell))
        outputs = outputs.squeeze(0)
        prediction = self.fc_out(outputs)
        return prediction, hidden, cell


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

        assert encoder.hidden_dim == decoder.hidden_dim, "Hidden dimensions of encoder and decoder must be equal!"
        assert encoder.num_layers == decoder.num_layers, "Encoder and decoder must have equal number of layers!"

    def forward(self, src, target, teacher_forcing_ratio=0.5):
        batch_size = target.shape[1]
        target_len = target.shape[0]
        target_vocab_size = self.decoder.output_dim
        outputs = torch.zeros(target_len, batch_size, target_vocab_size).to(self.device)
        hidden, cell = self.encoder(src)
        inputs = target[0, :]
        for t in range(1, target_len):
            # prediction: batch_size target_vocab_size
            prediction, hidden, cell = self.decoder(inputs, hidden, cell)
            outputs[t] = prediction
            teacher_force = random.random() < teacher_forcing_ratio
            top = prediction.argmax(1)
            inputs = target[t] if teacher_force else top
        return outputs


INPUT_DIM = len(SRC.vocab)
OUTPUT_DIM = len(TRG.vocab)
ENC_EMB_DIM = 256
DEC_EMB_DIM = 256
HID_DIM = 512
N_LAYERS = 2
ENC_DROPOUT = 0.5
DEC_DROPOUT = 0.5

enc = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT)
dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT)

model = Seq2Seq(enc, dec, device).to(device)


def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.uniform_(param.data, -0.08, 0.08)


model.apply(init_weights)


def count_parameters(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


print(f'The model has {count_parameters(model):,} trainable parameters')
optimizer = optim.Adam(model.parameters())

TRG_PAD_IDX = TRG.vocab.stoi[TRG.pad_token]
criterion = nn.CrossEntropyLoss(ignore_index=TRG_PAD_IDX)


def train(model, iterator, optimizer, criterion, clip):
    model.train()
    epoch_loss = 0
    for i, batch in enumerate(iterator):
        src = batch.src
        target = batch.trg
        optimizer.zero_grad()
        # output: target batch_size target_vocab_size
        output = model(src, target)
        output_dim = output.shape[-1]
        output = output[1:].view(-1, output_dim)
        target = target[1:].view(-1)
        loss = criterion(output, target)
        loss.backward()
        nn.utils.clip_grad_norm(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / len(iterator)


def evaluate(model, iterator, criterion):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            src = batch.src
            target = batch.trg
            output = model(src, target, 0)
            output_dim = output.shape[-1]
            output = output[1:].view(-1, output_dim)
            target = target[1:].view(-1)
            loss = criterion(output, target)
            epoch_loss += loss.item()
    return epoch_loss / len(iterator)


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_minutes = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_minutes * 60))
    return elapsed_minutes, elapsed_secs


N_EPOCHS = 10
CLIP = 1

best_valid_loss = float('inf')

for epoch in range(N_EPOCHS):
    start_time = time.time()
    train_loss = train(model, train_iterator, optimizer, criterion, CLIP)
    valid_loss = evaluate(model, valid_iterator, criterion)
    end_time = time.time()
    epoch_minutes, epoch_secs = epoch_time(start_time, end_time)

    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'tut1-model.pt')

    print(f'Epoch: {epoch + 1:02} | Time: {epoch_minutes}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')


model.load_state_dict(torch.load('tut1-model.pt'))
test_loss = evaluate(model, test_iterator, criterion)
print(f'| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |')
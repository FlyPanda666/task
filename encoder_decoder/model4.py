"""
Packed padded sequences are used to tell our RNN to skip over padding tokens in our encoder.
Masking explicitly forces the model to ignore certain values, such as attention over padded elements.
Both of these techniques are commonly used in NLP.
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
from torch.nn.modules.rnn import GRU
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


# When using packed padded sequences, we need to tell PyTorch how long the actual (non-padded) sequences are
SRC = Field(tokenize=tokenize_de,
            init_token='<sos>',
            eos_token='<eos>',
            lower=True,
            include_lengths=True)

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
# One quirk about packed padded sequences is that all elements in the batch need to be sorted by their non-padded
# lengths in descending order, i.e. the first sentence in the batch needs to be the longest.
# We use two arguments of the iterator to handle this,
# sort_within_batch which tells the iterator that the contents of the batch need to be sorted,
# and sort_key a function which tells the iterator how to sort the elements in the batch.
# Here, we sort by the length of the src sentence.
train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
    (train_data, valid_data, test_data),
    batch_size=BATCH_SIZE,
    sort_within_batch=True,
    sort_key=lambda x: len(x.src),
    device=device)

print(train_iterator.data()[0])
print(valid_iterator.data()[:20])
print(test_iterator.data()[:20])


class Encoder(nn.Module):
    def __init__(self, input_dim, embedding_dim, encoder_hidden_dim, decoder_hidden_dim, dropout, bidirectional=True):
        """
        :param input_dim: source的字典大小.
        :param embedding_dim: 词向量的维度.
        :param encoder_hidden_dim:
        :param decoder_hidden_dim:
        :param dropout:
        """
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=input_dim, embedding_dim=embedding_dim)
        self.rnn = GRU(input_size=embedding_dim, hidden_size=encoder_hidden_dim, bidirectional=bidirectional)
        self.fc = nn.Linear(encoder_hidden_dim * 2, decoder_hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_len):
        """The changes here all within the forward method.
        It now accepts the lengths of the source sentences as well as the sentences themselves.
        After the source sentence (padded automatically within the iterator) has been embedded,
        we can then use pack_padded_sequence on it with the lengths of the sentences.
        Note that the tensor containing the lengths of the sequences must be a CPU tensor as of the latest version of
        PyTorch, which we explicitly do so with to('cpu').
        :param src: [src, batch_size]
        :param src_len: [batch_size]
        :return:
        """
        embed = self.embedding(src)
        embed = self.dropout(embed)
        # need to explicitly put lengths on cpu!
        packed_embed = nn.utils.rnn.pack_padded_sequence(embed, lengths=src_len.to("cpu"))
        # outputs: [src, batch_size, hidden_dim*directions]
        # hidden: [num_layers*directions, batch_size, hidden_dim]
        # cell: [num_layers*directions, batch_size, hidden_dim]
        packed_outputs, hidden = self.rnn(packed_embed)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs)
        hidden = torch.tanh(self.fc(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)))
        return outputs, hidden


class Attention(nn.Module):
    """The layer will output an attention vector,
    that is the length of the source sentence, each element is between 0 and 1 and the entire vector sums to 1.

    Previously, we allowed this module to "pay attention" to padding tokens within the source sentence.
    However, using masking, we can force the attention to only be over non-padding elements.
    """
    def __init__(self, encoder_hidden_dim, decoder_hidden_dim):
        super().__init__()
        self.attention = nn.Linear(encoder_hidden_dim * 2 + decoder_hidden_dim, decoder_hidden_dim)
        self.v = nn.Linear(decoder_hidden_dim, 1, bias=False)

    def forward(self, encoder_outputs, hidden, mask):
        """this layer takes what we have decoded so far, and all of what we have encoded, to produce a vector,
        that represents which words in the source sentence we should pay the most attention to in order to
        correctly predict the next word to decode.

        First, we calculate the energy between the previous decoder hidden state and the encoder hidden states.
        As our encoder hidden states are a sequence of tensors, our previous decoder hidden state is a single tensor,
        the first thing we do is repeat the previous decoder hidden state t times. We then calculate the energy,
        between them by concatenating them together and passing them through a linear layer and a activation function.
        :param hidden: 是decoder以前各个时刻的输出.[batch_size, hidden_dim]
        :param encoder_outputs: 是encoder的输出.[src_len, batch_size, hidden_dim]
        :param mask: [batch_size, src_len], that is 1 when the source sentence token is not a padding token.
        :return:
        """
        source_len = encoder_outputs.shape[0]
        # hidden: [batch_size, decoder_hidden_dim] -> [batch_size, 1, decoder_hidden_dim] ->
        # [batch_size, source_len, decoder_hidden_dim]
        hidden = hidden.unsqueeze(1).repeat(1, source_len, 1)
        # encoder_outputs: [source_len, batch_size, encoder_hidden_dim * num_layers] ->
        # [batch_size, source_len, encoder_hidden_dim * num_layers]
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        # [batch_size, source_len, decoder_hidden_dim]
        # energy: [batch_size, source_len, decoder_hidden_dim]
        energy = torch.tanh(self.attention(torch.cat((hidden, encoder_outputs), dim=2)))
        # attention: [batch_size, source_len]
        attention = self.v(energy).squeeze(2)
        # We apply the mask after the attention has been calculated, but before it has been normalized by the softmax
        # function. It is applied using masked_fill.
        attention = attention.masked_fill(mask == 0, -1e10)
        return torch.softmax(attention, dim=1)


class Decoder(nn.Module):
    def __init__(self, output_dim, embedding_dim, encoder_hidden_dim, decoder_hidden_dim, dropout, attention):
        """
        :param output_dim:
        :param embedding_dim:
        :param encoder_hidden_dim:
        :param decoder_hidden_dim:
        :param dropout:
        :param attention:
        """
        super().__init__()
        self.attention = attention
        self.output_dim = output_dim
        self.embedding = nn.Embedding(num_embeddings=output_dim, embedding_dim=embedding_dim)
        self.rnn = GRU(input_size=embedding_dim + 2 * encoder_hidden_dim, hidden_size=decoder_hidden_dim)
        self.fc_out = nn.Linear(embedding_dim + 2 * encoder_hidden_dim + decoder_hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs, hidden, encoder_outputs, mask):
        """
        :param inputs: 当前时刻的输入,只有一个token.[batch_size]
        :param hidden: 隐状态.[batch_size, decoder_hidden_dim]
        :param encoder_outputs: 上下文.[src_len, batch_size, encoder_hidden_dim*2]
        :param mask: [batch_size, src_len]
        :return:
        """
        inputs = inputs.unsqueeze(0)
        inputs_embed = self.embedding(inputs)
        inputs_embed = self.dropout(inputs_embed)

        # inputs_embed: [1, batch_size, embedding_dim]
        a = self.attention(encoder_outputs, hidden, mask)
        a = a.unsqueeze(1)
        # a: [batch_size, 1, src_len]
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        # encoder_outputs: [batch_size, src_len, encoder_hidden_dim*2]
        weighted = torch.bmm(a, encoder_outputs)
        weighted = weighted.permute(1, 0, 2)
        # weighted: [1, batch_size, encoder_hidden_dim*2]
        rnn_input = torch.cat((inputs_embed, weighted), dim=2)
        outputs, hidden = self.rnn(rnn_input, hidden.unsqueeze(0))
        # outputs: [1, batch_size, decoder_hidden_dim]

        outputs = torch.cat((inputs_embed.squeeze(0), outputs.squeeze(0), weighted.squeeze(0)), dim=1)
        prediction = self.fc_out(outputs)
        return prediction, hidden.squeeze(0), a.squeeze(1)


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device, src_pad_idx):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.src_pad_idx = src_pad_idx

    def create_mask(self, src):
        mask = (src != self.src_pad_idx).long().permute(1, 0)
        return mask

    def forward(self, src, src_len, target, teacher_forcing_ratio=0.5):
        """
        :param src: [src_len, batch_size]
        :param src_len: [batch_size]
        :param target: [target_len, batch_size]
        :param teacher_forcing_ratio:
        :return:
        """
        batch_size = target.shape[1]
        target_len = target.shape[0]
        target_vocab_size = self.decoder.output_dim
        outputs = torch.zeros(target_len, batch_size, target_vocab_size).to(self.device)
        encoder_outputs, hidden = self.encoder(src, src_len)

        inputs = target[0, :]
        mask = self.create_mask(src)

        for t in range(1, target_len):
            # prediction: [batch_size, target_vocab_size]
            prediction, hidden, _ = self.decoder(inputs, hidden, encoder_outputs, mask)
            outputs[t] = prediction
            teacher_force = random.random() < teacher_forcing_ratio
            top = prediction.argmax(1)
            inputs = target[t] if teacher_force else top
        return outputs


INPUT_DIM = len(SRC.vocab)
OUTPUT_DIM = len(TRG.vocab)
ENC_EMB_DIM = 256
DEC_EMB_DIM = 256
ENC_HID_DIM = 512
DEC_HID_DIM = 512
ENC_DROPOUT = 0.5
DEC_DROPOUT = 0.5
SRC_PAD_IDX = SRC.vocab.stoi[SRC.pad_token]

attn = Attention(ENC_HID_DIM, DEC_HID_DIM)
enc = Encoder(INPUT_DIM, ENC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, ENC_DROPOUT)
dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, DEC_DROPOUT, attn)

model = Seq2Seq(enc, dec, SRC_PAD_IDX, device).to(device)


def init_weights(m):
    for name, param in m.named_parameters():
        if 'weight' in name:
            nn.init.normal_(param.data, mean=0, std=0.01)
        else:
            nn.init.constant_(param.data, 0)


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
        src, src_len = batch.src
        target = batch.trg
        optimizer.zero_grad()
        # output: target batch_size target_vocab_size
        output = model(src, src_len, target)
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
            src, src_len = batch.src
            target = batch.trg
            output = model(src, src_len, target, 0)
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


N_EPOCHS = 20
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
        torch.save(model.state_dict(), 'tut4-model.pt')

    print(f'Epoch: {epoch + 1:02} | Time: {epoch_minutes}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')


model.load_state_dict(torch.load('tut4-model.pt'))
test_loss = evaluate(model, test_iterator, criterion)
print(f'| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |')
from abc import ABC

import torch
import torch.nn as nn


class TextRNN(nn.Module, ABC):
    def __init__(self, args):
        super(TextRNN, self).__init__()
        self.hidden_size = args.hidden_size
        self.layer_num = args.layer_num
        self.bidirectional = args.bidirectional
        self.embedding = nn.Embedding(args.vocab_size, args.embedding_dim)
        if args.static:
            self.embedding = self.embedding.from_pretrained(args.vectors, freeze=not args.fine_tune)
        self.lstm = nn.LSTM(
            args.embedding_dim, self.hidden_size, self.layer_num, batch_first=True, bidirectional=self.bidirectional
        )
        if self.bidirectional:
            self.fc = nn.Linear(self.hidden_size * 2,  args.label_num)
        else:
            self.fc = nn.Linear(self.hidden_size, args.label_num)

    def forward(self, x):
        x = self.embedding(x)
        if self.bidirectional:
            h0 = torch.zeros(self.layer_num * 2, x.size(0), self.hidden_size)
            c0 = torch.zeros(self.layer_num * 2, x.size(0), self.hidden_size)
        else:
            h0 = torch.zeros(self.layer_num, x.size(0), self.hidden_size)
            c0 = torch.zeros(self.layer_num, x.size(0), self.hidden_size)

        out, (hn, cn) = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

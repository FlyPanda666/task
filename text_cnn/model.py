from abc import ABC

import torch
import torch.nn as nn
import torch.nn.functional as F


class TextCNN(nn.Module, ABC):
    def __init__(self, args):
        super(TextCNN, self).__init__()
        self.embedding = nn.Embedding(args.vocabulary_size, args.embedding_dim)
        self.convolutions = nn.ModuleList([
            nn.Conv2d(args.channel_num, args.kernel_number, (k, args.embedding_dim))
            for k in map(int, args.kernel_size.split(","))
        ])
        self.dropout = nn.Dropout(args.dropout)
        self.fc = nn.Linear(len(args.kernel_size) * args.kernel_number, args.class_num)

    def forward(self, x):
        x = self.embedding(x)
        x = x.unsqueeze(1)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convolutions]
        x = [F.max_pool1d(line, line.size(2)).squeeze(2) for line in x]
        x = torch.cat(x, 1)
        x = self.dropout(x)
        logit = self.fc(x)
        return logit

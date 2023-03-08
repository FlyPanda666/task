import torch
from torch.nn.modules.rnn import GRU

net = GRU(2, 5)
x = torch.rand(3, 4, 2)
outputs, hidden = net(x)
print(outputs.shape)
print(hidden.shape)

net2 = GRU(input_size=2, hidden_size=5, bidirectional=True)
outputs, hidden = net2(x)
print(outputs.shape)
print(hidden.shape)

net3 = GRU(input_size=2, hidden_size=5, bidirectional=True, num_layers=3)
outputs, hidden = net3(x)
print(outputs.shape)
print(hidden.shape)

print(outputs)
print(hidden)
print(hidden[-2, :, :])
print(hidden[-1, :, :])
print(outputs[-1, :, :5])
print(outputs[0, :, 5:])

print(outputs[0, None])  # shape: [1, 4, 10]
print(outputs[0, ...])  # shape: [4, 10]

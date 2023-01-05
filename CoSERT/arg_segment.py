import random
import torch
import numpy as np


def layer():
    hidden_states = torch.randn(2, 3, 4)
    all_hidden_states = ()
    all_attentions = ()
    for i in range(3):
        self_attention = torch.randn(2, 3, 4)
        cross_attention = torch.randn(2, 3, 4)
        all_hidden_states = all_hidden_states + (hidden_states,)
        hidden_states = torch.rand(2, 3, 4)
        all_attentions = all_attentions + (self_attention, cross_attention)

    x = tuple(v for v in [hidden_states, all_hidden_states,
                          all_attentions] if v is not None)

    print(type(x))
    print(len(x))
    print(type(x[0]))
    print(type(x[1]))
    print(type(x[2]))

    print(x[0].shape)
    for i in range(len(x[1])):
        print(x[1][i].shape)

    print(hidden_states)
    print(hidden_states[:, 0, :])
    print(hidden_states[:, 0])
    print(hidden_states[:, 0].equal(hidden_states[:, 0, :]))
    print(torch.ones(5, 4))
    print(torch.ones(5, 4).shape)
    attention_mask = torch.tensor([[[1, 2, 3], [3, 5, 1]], [[4, 3, 1], [9, 8, 6]]])
    extended_attention_mask = None
    if attention_mask.dim() == 3:
        extended_attention_mask = attention_mask[:, None, :, :]

    print(attention_mask)
    print(extended_attention_mask)
    print([None] * 10)


def position_shuffle():
    bsz = 3
    seq_len = 30
    attention_mask = torch.ones(bsz, 1, 1, seq_len)
    position_ids = torch.arange(512).expand((bsz, -1))[:, :seq_len]
    print(position_ids)

    shuffled_pid = []
    for bsz_id in range(bsz):
        sample_pid = position_ids[bsz_id]
        sample_mask = attention_mask[bsz_id]
        num_tokens = sample_mask.sum().int().item()
        indexes = list(range(num_tokens))
        import random
        random.shuffle(indexes)
        rest_indexes = list(range(num_tokens, seq_len))
        total_indexes = indexes + rest_indexes
        shuffled_pid.append(torch.index_select(
            sample_pid, 0, torch.tensor(total_indexes)).unsqueeze(0))
    print(shuffled_pid)
    print(torch.cat(shuffled_pid, 0))


def embedding_arg():
    bsz = 3
    seq_len = 30
    attention_mask = torch.ones(bsz, seq_len)
    position_ids = torch.arange(512).expand((bsz, -1))[:, :seq_len]
    print(position_ids)
    sample_rate = 0.2
    input_ids = torch.tensor([random.randint(3, 100) for _ in range(seq_len)])
    print(f"input_ids : {input_ids}")

    true_seq_len = attention_mask.sum(1).cpu().numpy()
    print(true_seq_len)
    mask = []
    for true_len in true_seq_len:
        print(f"true_len: {true_len}")
        sample_len = max(int(true_len * (1 - sample_rate)), 1)
        print(f"sample_len: {sample_len}")
        start_id = np.random.randint(0, high=true_len - sample_len + 1)
        tmp = [1] * seq_len
        for idx in range(start_id, start_id + sample_len):
            tmp[idx] = 0
        mask.append(tmp)
    print(f"mask : {mask}")
    mask = torch.ByteTensor(mask).bool()
    print(f"mask : {mask}")
    input_ids = input_ids.masked_fill(mask, value=0)

    attention_mask = attention_mask.masked_fill(mask, value=0)
    print(input_ids)
    print(attention_mask)


def embedding_replace():
    bsz = 3
    seq_len = 30
    hidden_size = 8
    noise_embedding = torch.randn(bsz, seq_len, hidden_size)
    embedding_output = torch.randn(bsz, seq_len, hidden_size)
    assert noise_embedding.shape == embedding_output.shape, (noise_embedding.shape, embedding_output.shape)
    print(noise_embedding.shape)


if __name__ == '__main__':
    embedding_replace()

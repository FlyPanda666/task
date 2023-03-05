from abc import ABC

import torch
import torch.nn as nn
from transformers import BertModel
from predefine_token import NONE
from data_load import idx2trigger, argument2idx, idx2argument
from utils_func import find_triggers


class Net(nn.Module, ABC):
    def __int__(self, trigger_size=None, argument_size=None, device=torch.device("cpu")):
        super(Net, self).__init__()
        self.bert = BertModel.from_pretrained("torch_roberta_wwm")
        self.hidden_size = 768
        self.rnn = nn.LSTM(bidirectional=True, num_layers=1, input_size=self.hidden_size,
                           hidden_size=self.hidden_size // 2, batch_first=True)
        self.linear_l = nn.Linear(self.hidden_size, self.hidden_size // 2)
        self.linear_r = nn.Linear(self.hidden_size, self.hidden_size // 2)
        self.fc_trigger = nn.Sequential(nn.Linear(self.hidden_size, trigger_size))
        self.fc_argument = nn.Sequential(nn.Linear(self.hidden_size, argument_size))
        self.device = device

    def predict_triggers(self, tokens_x_2d, mask, head_indexes_2d, arguments_2d=None, test=False):
        tokens_x_2d = torch.LongTensor(tokens_x_2d)
        mask = torch.LongTensor(mask)
        head_indexes_2d = torch.LongTensor(head_indexes_2d)

        if self.training:
            self.bert.train()
            encoded_layers = self.bert(input_ids=tokens_x_2d, attention_mask=mask)
            enc = encoded_layers.last_hidden_state
        else:
            self.bert.eval()
            with torch.no_grad():
                encoded_layers = self.bert(input_ids=tokens_x_2d, attention_mask=mask)
                enc = encoded_layers.last_hidden_state

        x = enc
        batch_size = tokens_x_2d.size()[0]
        for i in range(batch_size):
            x[i] = torch.index_select(x[i], 0, head_indexes_2d[i])

        trigger_logits = self.fc_trigger(x)
        trigger_hat_2d = trigger_logits.argmax(-1)
        x_rnn, h0, argument_candidate = [], [], []
        for i in range(batch_size):
            predicted_triggers = find_triggers([idx2trigger[trigger] for trigger in trigger_hat_2d[i].tolist()])
            for predicted_trigger in predicted_triggers:
                t_start, t_end, t_type_str = predicted_trigger
                event_tensor_l = self.linear_l(x[i, t_start, :])
                event_tensor_r = self.linear_r(x[i, t_end - 1, :])
                event_tensor = torch.stack([event_tensor_l, event_tensor_r])
                h0.append(event_tensor)
                x_rnn.append(x[i])
                argument_candidate.append((i, t_start, t_end, t_type_str))

        argument_logits, arguments_y_1d = [0], [0]
        argument_hat_2d = [{'events': {}} for _ in range(batch_size)]
        if len(argument_candidate) > 0:
            h0 = torch.stack(h0, dim=1)
            c0 = torch.zeros(h0.shape[:], dtype=torch.float)
            x_rnn = torch.stack(x_rnn)
            rnn_out, (hn, cn) = self.rnn(x_rnn, (h0, c0))
            argument_logits = self.fc_argument(rnn_out)
            argument_hat = argument_logits.argmax(-1)

            argument_hat_2d = [{'events': {}} for _ in range(batch_size)]
            for i in range(len(argument_hat)):
                ba, st, ed, event_type_str = argument_candidate[i]
                if (st, ed, event_type_str) not in argument_hat_2d[ba]['events']:
                    argument_hat_2d[ba]['events'][(st, ed, event_type_str)] = []
                predicted_arguments = find_triggers([idx2argument[argument] for argument in argument_hat[i].tolist()])
                for predicted_argument in predicted_arguments:
                    e_start, e_end, e_type_str = predicted_argument
                    argument_hat_2d[ba]['events'][(st, ed, event_type_str)].append((e_start, e_end, e_type_str))

            arguments_y_1d = []
            if not test:
                for i, t_start, t_end, t_type_str in argument_candidate:
                    a_label = [NONE] * x.shape[1]
                    if (t_start, t_end, t_type_str) in arguments_2d[i]['events']:
                        for (a_start, a_end, a_role) in arguments_2d[i]['events'][(t_start, t_end, t_type_str)]:
                            for j in range(a_start, a_end):
                                if j == a_start:
                                    a_label[j] = 'B-{}'.format(a_role)
                                else:
                                    a_label[j] = 'I-{}'.format(a_role)

                    a_label = [argument2idx[t] for t in a_label]
                    arguments_y_1d.append(a_label)

            arguments_y_1d = torch.LongTensor(arguments_y_1d)
        return trigger_logits, trigger_hat_2d, argument_logits, arguments_y_1d, argument_hat_2d

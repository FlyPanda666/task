import torch

from data_load import idx2trigger


def test(model, iterator, file_name):
    model.eval()
    words_all, triggers_hat_all, arguments_hat_all = [], [], []
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            tokens_x_2d, sample_id, sequence_length_1d, head_indexes_2d, mask, words_2d = batch
            trigger_logits, trigger_hat_2d, argument_logits, arguments_y_1d, argument_hat_2d = model.predict_triggers(
                tokens_x_2d=tokens_x_2d,
                mask=mask,
                head_indexes_2d=head_indexes_2d,
                test=True
            )
            words_all.extend(words_2d)
            triggers_hat_all.extend(trigger_hat_2d.cpu().numpy().tolist())
            arguments_hat_all.extend(argument_hat_2d)

    with open(file_name, mode="w") as file_out:
        for i, (words, triggers_hat, arguments_hat) in enumerate(zip(words_all, triggers_hat_all, arguments_hat_all)):
            triggers_hat = triggers_hat[:len(words) - 2]
            triggers_hat = [idx2trigger[hat] for hat in triggers_hat]
            for w, t_h in zip(words[1:-1], triggers_hat):
                file_out.write('{}\t{}\n'.format(w, t_h))
            tmp_events = {}
            if arguments_hat['events']:
                for k, v in arguments_hat['events'].items():
                    tmp_arg = [(''.join(words[1:-1][t[0]:t[1]]), t[2]) for t in v]
                    tmp_events[(''.join(words[1:-1][k[0]:k[1]]), k[2])] = tmp_arg
            arguments_hat['events'] = tmp_events
            print(''.join(words[1:-1]))
            print(arguments_hat['events'])

            file_out.write('{}\t#arguments#{}\n'.format(''.join(words[1:-1]), arguments_hat['events']))
            file_out.write("\n")

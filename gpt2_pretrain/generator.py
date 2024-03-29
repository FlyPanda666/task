import argparse
import os
from typing import List

import torch
import torch.nn.functional as F
from tqdm import trange
from transformers import GPT2LMHeadModel, BertTokenizer


def is_word(word):
    for item in list(word):
        if item not in 'qwertyuiopasdfghjklzxcvbnm':
            return False
    return True


def _is_chinese_char(char):
    cp = ord(char)
    if (
        (0x4E00 <= cp <= 0x9FFF) or (0x3400 <= cp <= 0x4DBF) or (0x20000 <= cp <= 0x2A6DF) or
            (0x2A700 <= cp <= 0x2B73F) or (0x2B740 <= cp <= 0x2B81F) or (0x2B820 <= cp <= 0x2CEAF) or
            (0xF900 <= cp <= 0xFAFF) or (0x2F800 <= cp <= 0x2FA1F)
    ):
        return True

    return False


def top_k_top_p_filtering(logits: torch.Tensor, top_k: int = 0, top_p: float = 0.0,
                          filter_value: float = -float('Inf')) -> torch.Tensor:
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
        From: https://arxiv.org/abs/1904.09751
        From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
        :param logits:
        :param filter_value:
        :param top_p:
        :param top_k:
    """
    assert logits.dim() == 1  # batch size 1 for now - could be updated for more but the code would be less clear
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
    return logits


def sample_sequence(model: GPT2LMHeadModel, context: List[int], length: int, n_ctx: int, tokenizer: BertTokenizer,
                    temperature: float = 1.0, top_k: int = 30, top_p: float = 0.0, repetition_penalty: float = 1.0,
                    device: str = 'cpu'):
    context = torch.tensor(context, dtype=torch.long, device=device)
    context = context.unsqueeze(0)
    generated = context
    with torch.no_grad():
        for _ in trange(length):
            # 一次生成一个，在末尾
            inputs = {'input_ids': generated[0][-(n_ctx - 1):].unsqueeze(0)}
            outputs = model(**inputs)
            # batch, seq_len, hidden_size 第二维为生成的，取最后一个
            next_token_logits = outputs[0][0, -1, :]
            for idx in set(generated):
                next_token_logits[idx] /= repetition_penalty
            next_token_logits = next_token_logits / temperature
            # 不取unk的词
            next_token_logits[tokenizer.convert_tokens_to_ids('[UNK]')] = -float('Inf')
            filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
            next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)
            generated = torch.cat((generated, next_token.unsqueeze(0)), dim=1)
    return generated.tolist()[0]


def fast_sample_sequence(model: GPT2LMHeadModel, context: List[int], length: int, temperature: float = 1.0,
                         top_k: int = 30, top_p: float = 0.0, device: str = 'cpu'):
    inputs = torch.LongTensor(context).view(1, -1).to(device)
    if len(context) > 1:
        _, past = model(inputs[:, :-1], None)[:2]
        prev = inputs[:, -1].view(1, -1)
    else:
        past = None
        prev = inputs
    generator = [] + context
    with torch.no_grad():
        for _ in trange(length):
            output = model(prev, past=past)
            output, past = output[:2]
            output = output[-1].squeeze(0) / temperature
            filtered_logits = top_k_top_p_filtering(output, top_k=top_k, top_p=top_p)
            next_token = torch.multinomial(torch.softmax(filtered_logits, dim=-1), num_samples=1)
            generator.append(next_token.item())
            prev = next_token.view(1, 1)
    return generator


# 通过命令行参数--fast_pattern，指定模式
def generate(n_ctx: int, model: GPT2LMHeadModel, context: List[int], length: int, tokenizer: BertTokenizer,
             temperature: int = 1, top_k: int = 0, top_p: float = 0.0, repetition_penalty: float = 1.0,
             device: str = 'cpu', is_fast_pattern: bool = False):
    if is_fast_pattern:
        return fast_sample_sequence(
            model, context, length, temperature=temperature, top_k=top_k, top_p=top_p, device=device)
    else:
        return sample_sequence(
            model, context, length, n_ctx, tokenizer=tokenizer, temperature=temperature, top_k=top_k, top_p=top_p,
            repetition_penalty=repetition_penalty, device=device)


def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='0,1,2,3', type=str, required=False, help='生成设备')
    parser.add_argument('--length', default=100, type=int, required=False, help='生成长度')
    parser.add_argument('--batch_size', default=1, type=int, required=False, help='生成的batch size')
    parser.add_argument('--n_samples', default=5, type=int, required=False, help='生成几个样本')
    parser.add_argument('--temperature', default=1, type=float, required=False, help='生成温度')
    parser.add_argument('--top_k', default=0, type=int, required=False, help='最高几选一')
    parser.add_argument('--top_p', default=0.9, type=float, required=False, help='最高积累概率')
    # 如果从头开始训练自己的模型,需要使用到下面的配置参数,主要包括模型的config文件以及vocab.txt文件.
    # parser.add_argument('--model_config', default='models/model_config.json', type=str, required=False, help='模型参数')
    # parser.add_argument('--tokenizer_path', default='models/vocab.txt', type=str, required=False, help='词表路径')
    # parser.add_argument('--model_path', default='models/c_news', type=str, required=False, help='模型路径')
    parser.add_argument('--no_word_piece', action='store_true', help='不做word piece切词')
    parser.add_argument('--segment', action='store_true', help='以词为单位')
    parser.add_argument('--fast_pattern', action='store_true', help='使用past加快生成速度')
    parser.add_argument('--save_samples', default=True, action='store_true', help='保存测试模型时产生的样本')
    parser.add_argument('--save_samples_path', default='.', type=str, required=False, help="保存样本的路径")
    parser.add_argument('--repetition_penalty', default=1.5, type=float, required=False, help="重复生成惩罚参数")
    # 这里是使用这个预训练好的中文模型进行further pretrain的,因此把上面的模型配置文件给屏蔽了.
    parser.add_argument('--further_pretrained_dir', default='./models/further_pretrained', type=str,
                        required=False, help='自己预训练好的模型路径')
    args = parser.parse_args()
    return args


def main():
    args = set_args()
    print('args:\n' + args.__repr__())
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = BertTokenizer.from_pretrained(args.further_pretrained_dir)
    model = GPT2LMHeadModel.from_pretrained(args.further_pretrained_dir)
    model.to(device)
    model.eval()
    n_ctx = model.config.n_ctx
    length = args.length
    if args.length == -1:
        length = model.config.n_ctx
    if args.save_samples:
        if not os.path.exists(args.save_samples_path):
            os.makedirs(args.save_samples_path)
    while True:
        title = input("请输入文章开头?\n")
        if len(title.strip()) == 0:
            continue
        raw_text = title.strip()
        context_tokens = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(raw_text))
        generated = 0
        for _ in range(args.n_samples // args.batch_size):
            out = generate(n_ctx=n_ctx, model=model, context=context_tokens, length=length,
                           is_fast_pattern=args.fast_pattern, tokenizer=tokenizer, temperature=args.temperature,
                           top_k=args.top_k, top_p=args.top_p, repetition_penalty=args.repetition_penalty,
                           device=device)
            for _ in range(args.batch_size):
                generated += 1
                text = tokenizer.convert_ids_to_tokens(out)
                for i, item in enumerate(text[:-1]):  # 确保英文前后有空格
                    if is_word(item) and is_word(text[i + 1]):
                        text[i] = item + ' '
                for i, item in enumerate(text):
                    if item == '[MASK]':
                        text[i] = ''
                    elif item == '[CLS]':
                        text[i] = '\n\n'
                    elif item == '[SEP]':
                        text[i] = '\n'
                info = "=" * 40 + title + ": SAMPLE " + str(generated) + " : " + "=" * 40 + "\n"
                print(info)
                text = ''.join(text).replace('##', '').strip()
                print(text)
                if args.save_samples:
                    with open(args.save_samples_path + '/samples.txt', 'a', encoding='utf8') as f:
                        f.write(info)
                        f.write(text)
                        f.write('\n')
                        f.write('=' * 90)
                        f.write('\n' * 2)
        print("=" * 80)


if __name__ == '__main__':
    main()

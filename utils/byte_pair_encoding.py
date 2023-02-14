from tokenizers.implementations.bert_wordpiece import BertWordPieceTokenizer
import pathlib

a = BertWordPieceTokenizer()
pt = "/Users/tal/task/utils/pg16457.txt"
a.train(pt)

path_to_ByteLevelBPE_tokenizer_pt_rep = pathlib.Path("./pretrained_tokenizer_bert/")
if not path_to_ByteLevelBPE_tokenizer_pt_rep.exists():
    path_to_ByteLevelBPE_tokenizer_pt_rep.mkdir(exist_ok=True, parents=True)
a.save_model(str(path_to_ByteLevelBPE_tokenizer_pt_rep))

exit()
import collections
import re


def get_vocab(filename):
    vocab = collections.defaultdict(int)
    with open(filename, 'r', encoding='utf-8') as content:
        for line in content:
            words = line.strip().split()
            for word in words:
                vocab[' '.join(list(word)) + ' </w>'] += 1
    return vocab


def get_stats(vocab):
    pairs = collections.defaultdict(int)
    for word, freq in vocab.items():
        symbols = word.split()
        for i in range(len(symbols) - 1):
            pairs[symbols[i], symbols[i + 1]] += freq
    return pairs


def merge_vocab(pair, v_in):
    v_out = {}
    bigram = re.escape(' '.join(pair))
    p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
    for word in v_in:
        w_out = p.sub(''.join(pair), word)
        v_out[w_out] = v_in[word]
    return v_out


def get_tokens(vocab):
    tokens = collections.defaultdict(int)
    for word, freq in vocab.items():
        word_tokens = word.split()
        for token in word_tokens:
            tokens[token] += freq
    return tokens


if __name__ == '__main__':
    vocab = {'l o w </w>': 5, 'l o w e r </w>': 2, 'n e w e s t </w>': 6, 'w i d e s t </w>': 3}
    # vocab = get_vocab('pg16457.txt')
    print('==========')
    print('Tokens Before BPE')
    tokens = get_tokens(vocab)
    print(tokens)
    print('Tokens: {}'.format(tokens))
    print('Number of tokens: {}'.format(len(tokens)))
    print('==========')

    num_merges = 1000
    for i in range(num_merges):
        pairs = get_stats(vocab)
        if not pairs:
            break
        best = max(pairs, key=pairs.get)
        vocab = merge_vocab(best, vocab)
        print('Iter: {}'.format(i))
        print('Best pair: {}'.format(best))
        tokens = get_tokens(vocab)
        print('Tokens: {}'.format(tokens))
        print('Number of tokens: {}'.format(len(tokens)))
        print('==========')

import argparse
import copy
import csv
import json
import logging
import random
import time
import torch
import transformers
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers.models.auto import AutoConfig, AutoModelWithLMHead, AutoTokenizer

logger = logging.getLogger(__name__)


def set_seed(seed: int):
    """设定随机种子.
    :param seed:
    :return:
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def add_task_specific_tokens(tokenizer):
    """修改tokenizer,主要是添加一些特殊的token.
    :param tokenizer:
    :return:
    """
    tokenizer.add_special_tokens({'additional_special_tokens': ['[T]', '[P]', '[Y]']})
    tokenizer.trigger_token = '[T]'
    tokenizer.trigger_token_id = tokenizer.convert_tokens_to_ids('[T]')
    tokenizer.predict_token = '[P]'
    tokenizer.predict_token_id = tokenizer.convert_tokens_to_ids('[P]')
    tokenizer.lama_y = '[Y]'
    tokenizer.lama_x_id = tokenizer.convert_tokens_to_ids('[Y]')


def load_pretrained(model_name):
    """加载预训练模型,包括tokenizer、model和config.
    :param model_name:
    :return:
    """
    config = AutoConfig.from_pretrained(model_name)
    model = AutoModelWithLMHead.from_pretrained(model_name)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name, add_prefix_space=True)
    add_task_specific_tokens(tokenizer)
    return config, model, tokenizer


def get_embeddings(model, config):
    """获取输入的文本的word embedding.
    :param model:
    :param config:
    :return:
    """
    base_model = getattr(model, config.model_type)
    embeddings = base_model.embeddings.word_embeddings
    return embeddings


class GradientStorage:
    """此类对象存储给定 PyTorch 模块输出的中间梯度，否则可能不会保留.
    """
    def __init__(self, module):
        self._stored_gradient = None
        module.register_backward_hook(self.hook)

    def hook(self, module, grad_in, grad_out):
        self._stored_gradient = grad_out[0]

    def get(self):
        return self._stored_gradient


def replace_trigger_tokens(model_inputs, trigger_ids, trigger_mask):
    """替换输入中的触发词.
    :param model_inputs:
    :param trigger_ids:
    :param trigger_mask:
    :return:
    """
    out = model_inputs.copy()
    input_ids = model_inputs['input_ids']
    trigger_ids = trigger_ids.repeat(trigger_mask.size(0), 1)
    try:
        filled = input_ids.masked_scatter(trigger_mask, trigger_ids)
    except RuntimeError:
        filled = input_ids
    out['input_ids'] = filled
    return out


class PredictWrapper:
    def __init__(self, model):
        self._model = model

    def __call__(self, model_inputs, trigger_ids):
        model_inputs = model_inputs.copy()
        trigger_mask = model_inputs.pop('trigger_mask')
        predict_mask = model_inputs.pop('predict_mask')
        model_inputs = replace_trigger_tokens(model_inputs, trigger_ids, trigger_mask)
        oo = self._model(**model_inputs)
        logits = oo.logits
        predict_logits = logits.masked_select(predict_mask.unsqueeze(-1)).view(logits.size(0), -1)
        return predict_logits


def encode_label(tokenizer, label, tokenize=False):
    """对标签进行编码.
    :param tokenizer:
    :param label:
    :param tokenize:
    :return:
    """
    encoded = None
    if isinstance(label, str):
        if tokenize:
            # Ensure label is properly tokenized, and only retain first token if it gets split into multiple tokens.
            tokens = tokenizer.tokenize(label)
            if len(tokens) > 1:
                raise ValueError(f'Label "{label}" gets mapped to multiple tokens.')
            if tokens[0] == tokenizer.unk_token:
                raise ValueError(f'Label "{label}" gets mapped to unk.')
            label = tokens[0]
        encoded = torch.tensor(tokenizer.convert_tokens_to_ids([label])).unsqueeze(0)
    elif isinstance(label, list):
        encoded = torch.tensor(tokenizer.convert_tokens_to_ids(label)).unsqueeze(0)
    elif isinstance(label, int):
        encoded = torch.tensor([[label]])
    return encoded


def get_loss(predict_logits, label_ids):
    """根据预测的概率和标签的id计算损失函数.
    :param predict_logits:
    :param label_ids:
    :return:
    """
    predict_logp = F.log_softmax(predict_logits, dim=-1)
    target_logp = predict_logp.gather(-1, label_ids)
    target_logp = target_logp - 1e32 * label_ids.eq(0)
    target_logp = torch.logsumexp(target_logp, dim=-1)
    return -target_logp


class AccuracyFn:
    def __init__(self, tokenizer, label_map, device, tokenize_labels=False):
        self._all_label_ids = []
        self._pred_to_label = []
        for label, label_tokens in label_map.items():
            self._all_label_ids.append(encode_label(tokenizer, label_tokens, tokenize_labels).to(device))
            self._pred_to_label.append(label)

    def __call__(self, predict_logits, gold_label_ids):
        gold_logp = get_loss(predict_logits, gold_label_ids)
        bsz = predict_logits.size(0)
        all_label_logp = []
        for label_ids in self._all_label_ids:
            label_logp = get_loss(predict_logits, label_ids.repeat(bsz, 1))
            all_label_logp.append(label_logp)
        all_label_logp = torch.stack(all_label_logp, dim=-1)
        ge_count = all_label_logp.le(gold_logp.unsqueeze(-1)).sum(-1)
        correct = ge_count.le(1)

        return correct.float()

    def predict(self, predict_logits):
        bsz = predict_logits.size(0)
        all_label_logp = []
        for label_ids in self._all_label_ids:
            label_logp = get_loss(predict_logits, label_ids.repeat(bsz, 1))
            all_label_logp.append(label_logp)
        all_label_logp = torch.stack(all_label_logp, dim=-1)
        _, predictions = all_label_logp.max(dim=-1)
        predictions = [self._pred_to_label[x] for x in predictions.tolist()]
        return predictions


def pad_squeeze_sequence(sequence, *args, **kwargs):
    return pad_sequence([x.squeeze(0) for x in sequence], *args, **kwargs)


class Collator:
    def __init__(self, pad_token_id=0):
        self._pad_token_id = pad_token_id

    def __call__(self, features):
        model_inputs, labels = list(zip(*features))
        proto_input = model_inputs[0]
        keys = list(proto_input.keys())
        padded_inputs = {}
        for key in keys:
            if key == 'input_ids':
                padding_value = self._pad_token_id
            else:
                padding_value = 0
            sequence = [x[key] for x in model_inputs]
            padded = pad_squeeze_sequence(sequence, batch_first=True, padding_value=padding_value)
            padded_inputs[key] = padded
        labels = pad_squeeze_sequence(labels, batch_first=True, padding_value=0)
        return padded_inputs, labels


def load_tsv(fname: str):
    """加载tsv格式的文件.
    :param fname:
    :return:
    """
    with open(fname, 'r') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            yield row


def load_jsonl(fname):
    """加载jsonl格式的文件.
    :param fname:
    :return:
    """
    with open(fname, 'r') as f:
        for line in f:
            yield json.loads(line)


LOADERS = {
    '.tsv': load_tsv,
    '.jsonl': load_jsonl
}
MAX_CONTEXT_LEN = 50


def load_trigger_dataset(fname, templatizer, use_ctx, limit=None):
    """
    :param fname:
    :param templatizer:
    :param use_ctx:
    :param limit:
    :return:
    """
    loader = LOADERS[fname.suffix]
    instances = []

    for x in loader(fname):
        try:
            if use_ctx:
                if 'evidences' not in x:
                    logger.warning('Skipping RE sample because it lacks context sentences: {}'.format(x))
                    continue
                evidences = x['evidences']

                # Randomly pick a context sentence
                obj_surface, masked_sent = random.choice(
                    [(evidence['obj_surface'], evidence['masked_sentence']) for evidence in evidences])
                words = masked_sent.split()
                if len(words) > MAX_CONTEXT_LEN:
                    masked_sent = ' '.join(words[:MAX_CONTEXT_LEN])

                # If truncated context sentence still has MASK, we need to replace it with object surface
                # We explicitly use [MASK] because all TREx fact's context sentences use it
                context = masked_sent.replace('[MASK]', obj_surface)
                x['context'] = context
                model_inputs, label_id = templatizer(x)
            else:
                model_inputs, label_id = templatizer(x)
        except ValueError as e:
            logger.warning('Encountered error "%s" when processing "%s".  Skipping.', e, x)
            continue
        else:
            instances.append((model_inputs, label_id))
    if limit:
        return random.sample(instances, limit)
    else:
        return instances


def load_augmented_trigger_dataset(fname, templatizer, limit=None):
    """
    :param fname:
    :param templatizer:
    :param limit:
    :return:
    """
    loader = LOADERS[fname.suffix]
    instances = []
    # For augmented relation extraction, we need to replace obj_label with another obj_label,
    # and replace obj_surface with a surface form of the new obj_label
    unique_objs_dict = defaultdict(list)
    # Also for augmented relation extraction, we need to accumulate all facts and process them afterwards
    facts = []

    for x in loader(fname):
        try:
            sub_label = x['sub_label']
            obj_label = x['obj_label']

            # For relation extraction, skip facts that don't have context sentence
            if 'evidences' not in x:
                logger.warning('Skipping RE sample because it lacks context sentences: {}'.format(x))
                continue

            evidences = x['evidences']

            # Gather all UNIQUE objects and their surface forms if its augmented relation extraction
            for evidence in evidences:
                obj_surface = evidence['obj_surface']
                masked_sent = evidence['masked_sentence']
                unique_objs_dict[obj_label].append(obj_surface)

            # Randomly pick a context sentence
            obj_surface, masked_sent = random.choice(
                [(evidence['obj_surface'], evidence['masked_sentence']) for evidence in evidences])
            words = masked_sent.split()
            if len(words) > MAX_CONTEXT_LEN:
                # If the masked sentence is too long, use the first X tokens.
                masked_sent = ' '.join(words[:MAX_CONTEXT_LEN])

            x['context'] = masked_sent
            facts.append(x)
        except ValueError as e:
            logger.warning('Encountered error "%s" when processing "%s".  Skipping.', e, x)

    # Go through all facts and replace each object with a new one.
    # Also insert the new object (surface form) into the masked sentence
    synth_facts = []
    for fact in facts:
        sub_label = fact['sub_label']
        obj_label = fact['obj_label']
        masked_sent = fact['context']
        # print('Original fact: ({}, {}, {})'.format(sub_label, obj_label, masked_sent))
        synth_obj_label = random.choice([x for x in unique_objs_dict.keys() if x != obj_label])
        synth_obj_surface = random.choice(unique_objs_dict[synth_obj_label])
        synth_ctx = masked_sent.replace('[MASK]', synth_obj_surface)
        # print('Synthetic fact: ({}, {}, {})\n'.format(sub_label, synth_obj_label, synth_ctx))
        # Reassign the labels and context sentence
        synth_fact = copy.deepcopy(fact)
        synth_fact['sub_label'] = sub_label
        synth_fact['obj_label'] = synth_obj_label
        synth_fact['context'] = synth_ctx
        synth_facts.append(synth_fact)

    # Go through facts, templatize each one, then append them to instances
    for fact in synth_facts:
        try:
            model_inputs, label_id = templatizer(fact)
            instances.append((model_inputs, label_id))
        except ValueError as e:
            print(e)

    if limit:
        return random.sample(instances, limit)
    else:
        return instances


def load_classification_dataset(fname, tokenizer, input_field_a, input_field_b=None, label_field='label',
                                label_map=None, limit=None):
    """
    :param fname:
    :param tokenizer:
    :param input_field_a:
    :param input_field_b:
    :param label_field:
    :param label_map:
    :param limit:
    :return:
    """
    instances = []
    label_map = label_map or {}
    loader = LOADERS[fname.suffix]
    for instance in loader(fname):
        logger.debug(instance)
        model_inputs = tokenizer.encode_plus(
            instance[input_field_a],
            instance[input_field_b] if input_field_b else None,
            add_special_tokens=True,
            # add_prefix_space=True,
            return_tensors='pt'
        )
        # logger.debug(model_inputs)
        label = instance[label_field]
        if label not in label_map:
            label_map[label] = len(label_map)
        label_id = label_map[label]
        label_id = torch.tensor([[label_id]])  # To make collator expectation
        # logger.debug(f'Label id: {label_id}')
        instances.append((model_inputs, label_id))
    if limit:
        instances = random.sample(instances, limit)
    return instances, label_map


class TriggerTemplatizer:
    """An object to facilitate creating transformers-friendly triggers inputs from a template.

    Parameters
    ==========
    template : str
        The template string, comprised of the following tokens:
            [T] to mark a trigger placeholder.
            [P] to mark a prediction placeholder.
            {fields} arbitrary fields instantiated from the dataset instances.
        For example a NLI template might look like:
            "[T] [T] [T] {premise} [P] {hypothesis}"
    tokenizer : PretrainedTokenizer
        A HuggingFace tokenizer. Must have special trigger and predict tokens.
    add_special_tokens : bool
        Whether or not to add special tokens when encoding. Default: False.
    """

    def __init__(self, template, config, tokenizer, label_field='label', label_map=None,
                 tokenize_labels=False, add_special_tokens=False, use_ctx=False):
        if not hasattr(tokenizer, 'predict_token') or not hasattr(tokenizer, 'trigger_token'):
            raise ValueError('Tokenizer missing special trigger and predict tokens in vocab.')
        self._template = template
        self._config = config
        self._tokenizer = tokenizer
        self._label_field = label_field
        self._label_map = label_map
        self._tokenize_labels = tokenize_labels
        self._add_special_tokens = add_special_tokens
        self._use_ctx = use_ctx

    @property
    def num_trigger_tokens(self):
        return sum(token == '[T]' for token in self._template.split())

    def __call__(self, format_kwargs):
        # Format the template string
        format_kwargs = format_kwargs.copy()
        label = format_kwargs.pop(self._label_field)
        text = self._template.format(**format_kwargs)
        if label is None:
            raise Exception(f'Bad data: {text}')

        # Have the tokenizer encode the text and process the output to:
        # - Create a trigger and predict mask
        # - Replace the predict token with a mask token
        model_inputs = self._tokenizer.encode_plus(
            text, add_special_tokens=self._add_special_tokens, return_tensors='pt')
        input_ids = model_inputs['input_ids']
        trigger_mask = input_ids.eq(self._tokenizer.trigger_token_id)
        predict_mask = input_ids.eq(self._tokenizer.predict_token_id)
        input_ids[predict_mask] = self._tokenizer.mask_token_id

        model_inputs['trigger_mask'] = trigger_mask
        model_inputs['predict_mask'] = predict_mask

        # For relation extraction with BERT, update token_type_ids to reflect the two different sequences.
        if self._use_ctx and self._config.model_type == 'bert':
            sep_token_indices = (input_ids.squeeze(0) == self._tokenizer.convert_tokens_to_ids(self._tokenizer.sep_token)).nonzero().flatten()
            sequence_b_indices = torch.arange(sep_token_indices[0], sep_token_indices[1] + 1).long().unsqueeze(0)
            model_inputs['token_type_ids'].scatter_(1, sequence_b_indices, 1)

        # Encode the label(s)
        if self._label_map is not None:
            label = self._label_map[label]
        label_id = encode_label(tokenizer=self._tokenizer, label=label, tokenize=self._tokenize_labels)
        return model_inputs, label_id


def isupper(idx, tokenizer):
    """Determines whether a token (e.g., word piece) begins with a capital letter.
    :param idx:
    :param tokenizer:
    :return:
    """
    _isupper = False
    # We only want to check tokens that begin words. Since byte-pair encoding
    # captures a prefix space, we need to check that the decoded token begins
    # with a space, and has a capitalized second character.
    if isinstance(tokenizer, transformers.GPT2Tokenizer):
        decoded = tokenizer.decode([idx])
        if decoded[0] == ' ' and decoded[1].isupper():
            _isupper = True
    # For all other tokenization schemes, we can just check the first character
    # is capitalized.
    elif tokenizer.decode([idx])[0].isupper():
        _isupper = True
    return _isupper


def hotflip_attack(averaged_grad, embedding_matrix, increase_loss=False, num_candidates=1, filter_=None):
    """Returns the top candidate replacements."""
    with torch.no_grad():
        gradient_dot_embedding_matrix = torch.matmul(embedding_matrix, averaged_grad)
        if filter_ is not None:
            gradient_dot_embedding_matrix -= filter_
        if not increase_loss:
            gradient_dot_embedding_matrix *= -1
        _, top_k_ids = gradient_dot_embedding_matrix.topk(num_candidates)

    return top_k_ids


def run_model(args):
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info('Loading model, tokenizer, etc.')
    config, model, tokenizer = load_pretrained(args.model_name)
    model.to(device)
    embeddings = get_embeddings(model, config)
    embedding_gradient: GradientStorage = GradientStorage(embeddings)
    predictor = PredictWrapper(model)

    if args.label_map is not None:
        label_map = json.loads(args.label_map)
        logger.info(f"Label map: {label_map}")
    else:
        label_map = None
        logger.info('No label map')

    templatizer = TriggerTemplatizer(args.template, config, tokenizer, label_map=label_map,
                                     label_field=args.label_field,
                                     tokenize_labels=args.tokenize_labels, add_special_tokens=False,
                                     use_ctx=args.use_ctx)

    # Obtain the initial trigger tokens and label mapping
    if args.initial_trigger:
        trigger_ids = tokenizer.convert_tokens_to_ids(args.initial_trigger)
        logger.debug(f'Initial trigger: {args.initial_trigger}')
        logger.debug(f'Trigger ids: {trigger_ids}')
        assert len(trigger_ids) == templatizer.num_trigger_tokens
    else:
        trigger_ids = [tokenizer.mask_token_id] * templatizer.num_trigger_tokens
    trigger_ids = torch.tensor(trigger_ids, device=device).unsqueeze(0)
    best_trigger_ids = trigger_ids.clone()

    # NOTE: Accuracy can only be computed if a fixed pool of labels is given, which currently
    # requires the label map to be specified. Since producing a label map may be cumbersome (e.g.,
    # for link prediction tasks), we just use (negative) loss as the evaluation metric in these cases.
    if label_map:
        evaluation_fn = AccuracyFn(tokenizer, label_map, device)
    else:
        def evaluation_fn(x, y):
            return -get_loss(x, y)

    logger.info('Loading datasets')
    collator = Collator(pad_token_id=tokenizer.pad_token_id)

    if args.perturbed:
        train_dataset = load_augmented_trigger_dataset(args.train, templatizer, limit=args.limit)
    else:
        train_dataset = load_trigger_dataset(args.train, templatizer, use_ctx=args.use_ctx, limit=args.limit)
    train_loader = DataLoader(train_dataset, batch_size=args.bsz, shuffle=True, collate_fn=collator)

    if args.perturbed:
        dev_dataset = load_augmented_trigger_dataset(args.dev, templatizer)
    else:
        dev_dataset = load_trigger_dataset(args.dev, templatizer, use_ctx=args.use_ctx)
    dev_loader = DataLoader(dev_dataset, batch_size=args.eval_size, shuffle=False, collate_fn=collator)

    # To "filter" unwanted trigger tokens, we subtract a huge number from their logits.
    filter = torch.zeros(tokenizer.vocab_size, dtype=torch.float32, device=device)
    if args.filter:
        logger.info('Filtering label tokens.')
        if label_map:
            for label_tokens in label_map.values():
                label_ids = encode_label(tokenizer, label_tokens).unsqueeze(0)
                filter[label_ids] = -1e32
        else:
            for _, label_ids in train_dataset:
                filter[label_ids] = -1e32
        logger.info('Filtering special tokens and capitalized words.')
        for word, idx in tokenizer.get_vocab().items():
            if len(word) == 1 or idx >= tokenizer.vocab_size:
                continue
            # Filter special tokens.
            if idx in tokenizer.all_special_ids:
                logger.debug('Filtered: %s', word)
                filter[idx] = -1e32
            # Filter capitalized words (lazy way to remove proper nouns).
            if isupper(idx, tokenizer):
                logger.debug('Filtered: %s', word)
                filter[idx] = -1e32

    logger.info('Evaluating')
    numerator = 0
    denominator = 0
    for model_inputs, labels in tqdm(dev_loader):
        model_inputs = {k: v.to(device) for k, v in model_inputs.items()}
        labels = labels.to(device)
        with torch.no_grad():
            logger.info(model_inputs)
            logger.info(trigger_ids)
            predict_logits = predictor(model_inputs, trigger_ids)
        numerator += evaluation_fn(predict_logits, labels).sum().item()
        denominator += labels.size(0)
    dev_metric = numerator / (denominator + 1e-13)
    logger.info(f'Dev metric: {dev_metric}')

    best_dev_metric = -float('inf')
    # Measure elapsed time of trigger search
    start = time.time()

    for i in range(args.iters):
        logger.info(f'Iteration: {i}')
        logger.info('Accumulating Gradient')
        model.zero_grad()

        pbar = tqdm(range(args.accumulation_steps))
        train_iter = iter(train_loader)
        averaged_grad = None

        # Accumulate
        for step in pbar:
            try:
                model_inputs, labels = next(train_iter)
            except:
                logger.warning(
                    'Insufficient data for number of accumulation steps. '
                    'Effective batch size will be smaller than specified.'
                )
                break
            model_inputs = {k: v.to(device) for k, v in model_inputs.items()}
            labels = labels.to(device)
            predict_logits = predictor(model_inputs, trigger_ids)
            loss = get_loss(predict_logits, labels).mean()
            loss.backward()

            grad = embedding_gradient.get()
            bsz, _, emb_dim = grad.size()
            selection_mask = model_inputs['trigger_mask'].unsqueeze(-1)
            grad = torch.masked_select(grad, selection_mask)
            grad = grad.view(bsz, templatizer.num_trigger_tokens, emb_dim)

            if averaged_grad is None:
                averaged_grad = grad.sum(dim=0) / args.accumulation_steps
            else:
                averaged_grad += grad.sum(dim=0) / args.accumulation_steps

        logger.info('Evaluating Candidates')
        pbar = tqdm(range(args.accumulation_steps))
        train_iter = iter(train_loader)

        token_to_flip = random.randrange(templatizer.num_trigger_tokens)
        candidates = hotflip_attack(averaged_grad[token_to_flip],
                                    embeddings.weight,
                                    increase_loss=False,
                                    num_candidates=args.num_cand,
                                    filter=filter)

        current_score = 0
        candidate_scores = torch.zeros(args.num_cand, device=device)
        denom = 0
        for step in pbar:
            try:
                model_inputs, labels = next(train_iter)
            except:
                logger.warning(
                    'Insufficient data for number of accumulation steps. '
                    'Effective batch size will be smaller than specified.'
                )
                break
            model_inputs = {k: v.to(device) for k, v in model_inputs.items()}
            labels = labels.to(device)
            with torch.no_grad():
                predict_logits = predictor(model_inputs, trigger_ids)
                eval_metric = evaluation_fn(predict_logits, labels)

            # Update current score
            current_score += eval_metric.sum()
            denom += labels.size(0)

            #Instead of iterating over tokens to flip we randomly change just one each time so the gradients don't get stale.
            for i, candidate in enumerate(candidates):
                # if candidate.item() in filter_candidates:
                #     candidate_scores[i] = -1e32
                #     continue

                temp_trigger = trigger_ids.clone()
                temp_trigger[:, token_to_flip] = candidate
                with torch.no_grad():
                    predict_logits = predictor(model_inputs, temp_trigger)
                    eval_metric = evaluation_fn(predict_logits, labels)

                candidate_scores[i] += eval_metric.sum()

        # TODO: Something cleaner. LAMA templates can't have mask tokens, so if
        # there are still mask tokens in the trigger then set the current score
        # to -inf.
        if args.print_lama:
            if trigger_ids.eq(tokenizer.mask_token_id).any():
                current_score = float('-inf')

        if (candidate_scores > current_score).any():
            logger.info('Better trigger detected.')
            best_candidate_score = candidate_scores.max()
            best_candidate_idx = candidate_scores.argmax()
            trigger_ids[:, token_to_flip] = candidates[best_candidate_idx]
            logger.info(f'Train metric: {best_candidate_score / (denom + 1e-13): 0.4f}')
        else:
            logger.info('No improvement detected. Skipping evaluation.')
            continue

        logger.info('Evaluating')
        numerator = 0
        denominator = 0
        for model_inputs, labels in tqdm(dev_loader):
            model_inputs = {k: v.to(device) for k, v in model_inputs.items()}
            labels = labels.to(device)
            with torch.no_grad():
                predict_logits = predictor(model_inputs, trigger_ids)
            numerator += evaluation_fn(predict_logits, labels).sum().item()
            denominator += labels.size(0)
        dev_metric = numerator / (denominator + 1e-13)

        logger.info(f'Trigger tokens: {tokenizer.convert_ids_to_tokens(trigger_ids.squeeze(0))}')
        logger.info(f'Dev metric: {dev_metric}')

        # TODO: Something cleaner. LAMA templates can't have mask tokens, so if
        # there are still mask tokens in the trigger then set the current score to -inf.
        if args.print_lama:
            if best_trigger_ids.eq(tokenizer.mask_token_id).any():
                best_dev_metric = float('-inf')

        if dev_metric > best_dev_metric:
            logger.info('Best performance so far')
            best_trigger_ids = trigger_ids.clone()
            best_dev_metric = dev_metric

    best_trigger_tokens = tokenizer.convert_ids_to_tokens(
        best_trigger_ids.squeeze(0))
    logger.info(f'Best tokens: {best_trigger_tokens}')
    logger.info(f'Best dev metric: {best_dev_metric}')
    if args.print_lama:
        # Templatize with [X] and [Y]
        if args.use_ctx:
            model_inputs, label_ids = templatizer({
                'sub_label': '[X]',
                'obj_label': tokenizer.lama_y,
                'context': ''
            })
        else:
            model_inputs, label_ids = templatizer({
                'sub_label': '[X]',
                'obj_label': tokenizer.lama_y,
            })
        lama_template = model_inputs['input_ids']
        # Instantiate trigger tokens
        lama_template.masked_scatter_(
            mask=model_inputs['trigger_mask'],
            source=best_trigger_ids.cpu())
        # Instantiate label token
        lama_template.masked_scatter_(
            mask=model_inputs['predict_mask'],
            source=label_ids)
        # Print LAMA JSON template
        relation = args.train.parent.stem

        # The following block of code is a bit hacky but whatever, it gets the job done
        if args.use_ctx:
            template = tokenizer.decode(lama_template.squeeze(
                0)[1:-1]).replace('[SEP] ', '').replace('</s> ', '').replace('[ X ]', '[X]')
        else:
            template = tokenizer.decode(lama_template.squeeze(0)[1:-1]).replace('[ X ]', '[X]')

        out = {
            'relation': args.train.parent.stem,
            'template': template
        }
        print(json.dumps(out))


def init_parse():
    logger.info("good...")
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=Path, required=True, help='Train data path')
    parser.add_argument('--dev', type=Path, required=True, help='Dev data path')
    parser.add_argument('--template', type=str, help='Template string')
    parser.add_argument('--label-map', type=str, default=None, help='JSON object defining label map')
    # LAMA-specific
    parser.add_argument('--tokenize-labels', action='store_true',
                        help='If specified labels are split into word pieces. Needed for LAMA probe experiments.')
    parser.add_argument('--filter', action='store_true',
                        help='If specified, filter out special tokens and gold objects.'
                             'Furthermore, tokens starting with capital '
                             'letters will not appear in triggers. Lazy approach for removing proper nouns.')
    parser.add_argument('--print-lama', action='store_true', help='Prints best trigger in LAMA format.')
    parser.add_argument('--initial-trigger', nargs='+', type=str, default=None, help='Manual prompt')
    parser.add_argument('--label-field', type=str, default='label', help='Name of the label field')
    parser.add_argument('--bsz', type=int, default=32, help='Batch size')
    parser.add_argument('--eval-size', type=int, default=256, help='Eval size')
    parser.add_argument('--iters', type=int, default=100, help='Number of iterations to run trigger search algorithm')
    parser.add_argument('--accumulation-steps', type=int, default=10)
    parser.add_argument('--model-name', type=str, default='bert-base-cased',
                        help='Model name passed to HuggingFace AutoX classes.')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--use-ctx', action='store_true', help='Use context sentences for relation extraction only')
    parser.add_argument('--perturbed', action='store_true',
                        help='Perturbed sentence evaluation of relation extraction: '
                             'replace each object in dataset with a random other object')
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--num-cand', type=int, default=10)
    parser.add_argument('--sentence-size', type=int, default=50)
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    if args.debug:
        level = logging.DEBUG
    else:
        level = logging.INFO
    logging.basicConfig(level=level)

    run_model(args)


if __name__ == '__main__':
    init_parse()

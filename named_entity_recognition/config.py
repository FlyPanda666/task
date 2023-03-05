import os
import torch
from utils.common.directory import Route

model_name = "bert_crf"
route = Route()
print(route.get_root())
data_dir = os.getcwd() + '/data/clue/'
print(data_dir)
train_dir = data_dir + 'train.npz'
test_dir = data_dir + 'test.npz'
files = ['train', 'test']
bert_model = route.get_bert_base_model()
roberta_model = route.get_roberta_wwm_large_ext()
model_dir = os.getcwd() + '/experiments/clue/'
log_dir = model_dir + 'train.log'
case_dir = os.getcwd() + '/case/bad_case.txt'

# 训练集、验证集划分比例
dev_split_size = 0.1

# 是否加载训练好的NER模型
load_before = False
hidden_dropout_prob = 0.01
# 是否对整个BERT进行fine tuning
full_fine_tuning = True

# hyper-parameter
learning_rate = 3e-5
weight_decay = 0.01
clip_grad = 5

batch_size = 32
epoch_num = 50
min_epoch_num = 5
patience = 0.0002
patience_num = 10

gpu = ''

if gpu != '':
    device = torch.device("cuda:{gpu}")
else:
    device = torch.device("cpu")


def get_labels():
    labels = ['keyword', 'aspect', 'influence']
    return labels


def labels_mapping():
    _labels = get_labels()
    _label2id = {"O": 0}
    for tag in ["B", "I", "S"]:
        for label in _labels:
            _label2id["{}-{}".format(tag, label)] = len(_label2id)

    _id2label = {ind: label for label, ind in _label2id.items()}
    return _label2id, _id2label


labels = get_labels()
label2id, id2label = labels_mapping()

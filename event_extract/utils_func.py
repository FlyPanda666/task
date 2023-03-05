from predefine_token import NONE, PAD


def build_vocabulary(labels: list, tagging_type="BIO") -> tuple:
    """ 构建字典函数。
    :param labels: 需要构建字典的序列列表。
    :param tagging_type: 采用的编码方式。
    :return:
    """
    all_labels = [NONE, PAD]
    for label in labels:
        if tagging_type:
            all_labels.append("B-{}".format(label))
            all_labels.append("I-{}".format(label))
        else:
            all_labels.append(label)
    label2idx = {label: index for index, label in enumerate(all_labels)}
    idx2label = {index: label for index, label in enumerate(all_labels)}
    return all_labels, label2idx, idx2label


def find_triggers(labels: list) -> list:
    """
    :param labels:
    :return:
    """
    result = []
    labels = [label.split("-") for label in labels]
    for i in range(len(labels)):
        if labels[i][0] == "B":
            result.append([i, i + 1, labels[i][1]])
    for item in result:
        j = item[1]
        while j < len(labels):
            if labels[j][0] == "I":
                j += 1
                item[1] = j
            else:
                break

    return [tuple(item) for item in result]


def calc_metric(y_true, y_predict):
    """ 计算模型得分。
    :param y_true:
    :param y_predict:
    :return:
    """
    num_proposed = len(y_predict)
    num_gold = len(y_true)
    y_true_set = set(y_true)
    num_correct = 0
    for item in y_predict:
        if item in y_true_set:
            num_correct += 1
    print('proposed: {}\tcorrect: {}\tgold: {}'.format(num_proposed, num_correct, num_gold))

    if num_proposed != 0:
        precision = num_correct / num_proposed
    else:
        precision = 1.0

    if num_gold != 0:
        recall = num_correct / num_gold
    else:
        recall = 1.0

    if precision + recall != 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0

    return precision, recall, f1

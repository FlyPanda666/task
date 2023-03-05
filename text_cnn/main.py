from load_data import load_data
import torch
from text_cnn.model import TextCNN
from utils.common.my_logger import logger
from argument import parse_arguments, train, test, predict


if __name__ == '__main__':
    args = parse_arguments()
    logger().info("Parameters:")

    train_iter, test_iter, text_field, label_field = load_data(args)

    cnn = TextCNN(args)
    if args.snapshot is not None:
        print('Loading model from {}...'.format(args.snapshot))
        cnn.load_state_dict(torch.load(args.snapshot))
    pytorch_total_params = sum(p.numel() for p in cnn.parameters() if p.requires_grad)
    print("Model parameters: " + str(pytorch_total_params))
    for attr, value in sorted(args.__dict__.items()):
        logger().info("{}={}".format(attr.upper(), value))
    if args.cuda:
        cnn = cnn.cuda()

    # step4：模型训练，验证和预测。
    if args.train:
        print("start training...")
        train(train_iter, test_iter, cnn, args)

    if args.test:
        print("start testing...")
        test(test_iter, cnn, args)

    if args.predict:
        print("start predicting...")
        while True:
            text = input(">>")
            label = predict(text, cnn, text_field, label_field, False)
            print(str(label) + " | " + text)

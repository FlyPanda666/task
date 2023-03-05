from text_rnn.model import TextRNN
from argument import parse_arguments, train, predict
from load_data import load_data


if __name__ == '__main__':
    args = parse_arguments()
    train_iteration, val_iteration, text_field, label_field = load_data(args)
    my_model = TextRNN(args)
    if args.train:
        train(train_iteration, val_iteration, my_model, args)
    if args.predict:
        print("start predicting...")
        while True:
            text = input(">>")
            label = predict(text, my_model, text_field, label_field, False)
            print(str(label) + " | " + text)

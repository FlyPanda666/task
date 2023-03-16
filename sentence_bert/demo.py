from sentence_transformers import SentenceTransformer, SentencesDataset, InputExample, evaluation, losses, models
from torch.utils.data import DataLoader


def read_data(data_dir: str):
    contents = []
    with open(data_dir, mode="r", encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines:
            item = line.replace("\n", "").split("\t")
            contents.append(item[0])
    return contents


def load_train_data():
    train_data = []
    train_content = read_data("")
    for idx in range(len(train_content)):
        train_data.append(InputExample(texts=[train_content[idx], train_content[idx]]))
    return train_data


model = SentenceTransformer('cyclone/simcse-chinese-roberta-wwm-ext')
data = load_train_data()
train_dataset = SentencesDataset(data, model)
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=32)

print(model)
train_loss = losses.MultipleNegativesRankingLoss(model)
model.fit(train_objectives=[(train_dataloader, train_loss)], show_progress_bar=True, epochs=3, warmup_steps=500, output_path="./save_model")

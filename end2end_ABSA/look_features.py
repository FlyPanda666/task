import torch


cached_features_file = "./data/laptop14/cached_dev_bert-base-uncased_128_laptop14"
features = torch.load(cached_features_file)
print(features[0])

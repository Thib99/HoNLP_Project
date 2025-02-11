from datasets import load_dataset

# Load WikiLingua dataset
dataset = load_dataset("GEM/wiki_lingua", "en")           # English articles -> English summaries
dataset_fr = load_dataset("GEM/wiki_lingua", "fr")        # French articles -> French summaries
dataset_cross = load_dataset("GEM/wiki_lingua", "fr_en")  # French articles -> English summaries

# Sample only articles with <= 512 tokens
max_length = 512
def dataset_sample(dataset):
    return dataset.filter(lambda example: len(example["source"]) <= max_length)

train = dataset_sample(dataset["train"])
val = dataset_sample(dataset["validation"])
test = dataset_sample(dataset["test"])

train_fr = dataset_sample(dataset_fr["train"])
val_fr = dataset_sample(dataset_fr["validation"])
test_fr = dataset_sample(dataset_fr["test"])

train_cross = dataset_sample(dataset_cross["train"])
val_cross = dataset_sample(dataset_cross["validation"])
test_cross = dataset_sample(dataset_cross["test"])

# To csv files
train.to_csv("train.csv")
val.to_csv("val.csv")
test.to_csv("test.csv")

train_fr.to_csv("train_fr.csv")
val_fr.to_csv("val_fr.csv")
test_fr.to_csv("test_fr.csv")

train_cross.to_csv("train_cross.csv")
val_cross.to_csv("val_cross.csv")
test_cross.to_csv("test_cross.csv")
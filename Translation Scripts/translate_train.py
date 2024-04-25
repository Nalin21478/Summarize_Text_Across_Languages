from datasets import load_dataset, Dataset
from easynmt import EasyNMT
import json
import torch
# Check if GPU is available
import torch
from datasets import load_dataset, Dataset
from easynmt import EasyNMT
import json

# Check if GPU is available
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# Initialize EasyNMT model on the available device
model = EasyNMT('opus-mt', device=device)

# Load dataset
dataset = load_dataset("hungnm/multilingual-amazon-review-sentiment-processed")
desired_features = ['stars', 'text', 'language', 'label']
reduced_dataset = dataset.select_columns(desired_features)
train = reduced_dataset['train']

# Filter datasets by language
de_train = train.filter(lambda example: example['language'] == 'de')
fr_train = train.filter(lambda example: example['language'] == 'fr')
es_train = train.filter(lambda example: example['language'] == 'es')
ja_train = train.filter(lambda example: example['language'] == 'ja')
zh_train = train.filter(lambda example: example['language'] == 'zh')

# Choose 50,000 data points for each language
de_train = de_train.select(range(20000))
fr_train = fr_train.select(range(20000))
es_train = es_train.select(range(20000))
ja_train = ja_train.select(range(20000))
zh_train = zh_train.select(range(20000))


# Group datasets by language
ds={"de":de_train,"fr":fr_train,"es":es_train, "ja":ja_train,"zh":zh_train}
language_dataset={"de":[],"es":[],"fr":[],"ja":[],"zh":[]}


# Translate and extend language datasets
for language,dset in ds.items():
    input_sentences = dset['text']
    translated_texts = model.translate(input_sentences, source_lang=language, target_lang='en', show_progress_bar=True)
    com_data = Dataset.from_dict({
        'text':dset['text'],
        'en': translated_texts,
        'label': dset['label'],
        'stars': dset['stars']
    })
    language_dataset[language].extend(com_data)
    
for i,j in language_dataset.items():
    file_name=f'{i}_train_en.json'
    with open(file_name, "w") as file:
        json.dump(j, file)

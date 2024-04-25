import json






# %%
import json

combined_data_test_de = []

# Load data from each JSON file, extract 'en' and 'stars', and append them to the combined_data list
files = [
    "/home/pavit21178/misc/Nalin_NLP/translated_test_jsons/de_test_en.json",
    # "/home/pavit21178/misc/Nalin_NLP/translated_test_jsons/es_test_en.json",
    # "/home/pavit21178/misc/Nalin_NLP/translated_test_jsons/fr_test_en.json",
    # "/home/pavit21178/misc/Nalin_NLP/translated_test_jsons/ja_test_en.json",
    # "/home/pavit21178/misc/Nalin_NLP/translated_test_jsons/zh_test_en.json"
]

for file_path in files:
    with open(file_path) as f:
        data = json.load(f)
        for item in data:
            combined_data_test_de.append({'en': item['en'], 'stars': item['stars']})

combined_data_test_es = []

# Load data from each JSON file, extract 'en' and 'stars', and append them to the combined_data list
files = [
    # "/home/pavit21178/misc/Nalin_NLP/translated_test_jsons/de_test_en.json",
    "/home/pavit21178/misc/Nalin_NLP/translated_test_jsons/es_test_en.json",
    # "/home/pavit21178/misc/Nalin_NLP/translated_test_jsons/fr_test_en.json",
    # "/home/pavit21178/misc/Nalin_NLP/translated_test_jsons/ja_test_en.json",
    # "/home/pavit21178/misc/Nalin_NLP/translated_test_jsons/zh_test_en.json"
]

for file_path in files:
    with open(file_path) as f:
        data = json.load(f)
        for item in data:
            combined_data_test_es.append({'en': item['en'], 'stars': item['stars']})
            
combined_data_test_fr = []

# Load data from each JSON file, extract 'en' and 'stars', and append them to the combined_data list
files = [
    # "/home/pavit21178/misc/Nalin_NLP/translated_test_jsons/de_test_en.json",
    # "/home/pavit21178/misc/Nalin_NLP/translated_test_jsons/es_test_en.json",
    "/home/pavit21178/misc/Nalin_NLP/translated_test_jsons/fr_test_en.json",
    # "/home/pavit21178/misc/Nalin_NLP/translated_test_jsons/ja_test_en.json",
    # "/home/pavit21178/misc/Nalin_NLP/translated_test_jsons/zh_test_en.json"
]

for file_path in files:
    with open(file_path) as f:
        data = json.load(f)
        for item in data:
            combined_data_test_fr.append({'en': item['en'], 'stars': item['stars']})
            

combined_data_test_ja = []

# Load data from each JSON file, extract 'en' and 'stars', and append them to the combined_data list
files = [
    # "/home/pavit21178/misc/Nalin_NLP/translated_test_jsons/de_test_en.json",
    # "/home/pavit21178/misc/Nalin_NLP/translated_test_jsons/es_test_en.json",
    # "/home/pavit21178/misc/Nalin_NLP/translated_test_jsons/fr_test_en.json",
    "/home/pavit21178/misc/Nalin_NLP/translated_test_jsons/ja_test_en.json",
    # "/home/pavit21178/misc/Nalin_NLP/translated_test_jsons/zh_test_en.json"
]

for file_path in files:
    with open(file_path) as f:
        data = json.load(f)
        for item in data:
            combined_data_test_ja.append({'en': item['en'], 'stars': item['stars']})
            

combined_data_test_zh = []

# Load data from each JSON file, extract 'en' and 'stars', and append them to the combined_data list
files = [
    # "/home/pavit21178/misc/Nalin_NLP/translated_test_jsons/de_test_en.json",
    # "/home/pavit21178/misc/Nalin_NLP/translated_test_jsons/es_test_en.json",
    # "/home/pavit21178/misc/Nalin_NLP/translated_test_jsons/fr_test_en.json",
    # "/home/pavit21178/misc/Nalin_NLP/translated_test_jsons/ja_test_en.json",
    "/home/pavit21178/misc/Nalin_NLP/translated_test_jsons/zh_test_en.json"
]

for file_path in files:
    with open(file_path) as f:
        data = json.load(f)
        for item in data:
            combined_data_test_zh.append({'en': item['en'], 'stars': item['stars']})


def remap_labels(example):
    class_mapping = {1: 0, 2: 1, 4: 2, 5: 3}
    example['stars'] = class_mapping.get(example['stars'])  
    return example


combined_data_test_de = [remap_labels(example) for example in combined_data_test_de]

combined_data_test_fr = [remap_labels(example) for example in combined_data_test_fr]

combined_data_test_es = [remap_labels(example) for example in combined_data_test_es]

combined_data_test_ja = [remap_labels(example) for example in combined_data_test_ja]

combined_data_test_zh = [remap_labels(example) for example in combined_data_test_zh]
# %%

# %%
from torch.utils.data import Dataset
import torch
class CustomDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        # Tokenize text
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Extract input IDs and attention masks
        input_ids = encoding['input_ids'].flatten()
        attention_mask = encoding['attention_mask'].flatten()
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': torch.tensor(label, dtype=torch.long)
        }


# %%


# %%


# %%
from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
max_length = 512


import torch
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, accuracy_score
from tqdm import tqdm

def evaluate_test_set(test_dataset, model, device):
    test_loader = DataLoader(test_dataset, batch_size=8*2, shuffle=True)
    
    test_predicted = []
    test_labels = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing", unit="batch"):
            inputs = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=inputs, attention_mask=attention_mask)
            logits = outputs.logits

            _, predicted = torch.max(logits, 1)
            test_predicted.extend(predicted.tolist())
            test_labels.extend(labels.tolist())

    test_f1 = f1_score(test_labels, test_predicted, average='weighted')
    accuracy = accuracy_score(test_labels, test_predicted)

    # Print test F1 score and accuracy
    print(f"Test F1 Score: {test_f1}")
    print(f"Test Accuracy: {accuracy}")

    return test_f1, accuracy


tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base')
device = torch.device('cuda')

test_dataset_de = CustomDataset([item['en'] for item in combined_data_test_de], [item['stars'] for item in combined_data_test_de], tokenizer, max_length)
test_loader_de = DataLoader(test_dataset_de, batch_size=64, shuffle=True)

test_dataset_fr = CustomDataset([item['en'] for item in combined_data_test_fr], [item['stars'] for item in combined_data_test_fr], tokenizer, max_length)
test_loader_fr = DataLoader(test_dataset_fr, batch_size=64, shuffle=True)

test_dataset_es = CustomDataset([item['en'] for item in combined_data_test_es], [item['stars'] for item in combined_data_test_es], tokenizer, max_length)
test_loader_es = DataLoader(test_dataset_es, batch_size=64, shuffle=True)

test_dataset_ja = CustomDataset([item['en'] for item in combined_data_test_ja], [item['stars'] for item in combined_data_test_ja], tokenizer, max_length)
test_loader_ja = DataLoader(test_dataset_ja, batch_size=64, shuffle=True)

test_dataset_zh = CustomDataset([item['en'] for item in combined_data_test_zh], [item['stars'] for item in combined_data_test_zh], tokenizer, max_length)
test_loader_zh = DataLoader(test_dataset_zh, batch_size=64, shuffle=True)

model = XLMRobertaForSequenceClassification.from_pretrained('/home/pavit21178/misc/Nalin_NLP/translated_xlm-robert_model_stars').to(device)

test_f1_de, accuracy_de = evaluate_test_set(test_dataset_de, model, device)
print("DE: ", test_f1_de, accuracy_de)

test_f1_fr, accuracy_fr = evaluate_test_set(test_dataset_fr, model, device)
print("fr: ", test_f1_fr, accuracy_fr)

test_f1_ja, accuracy_ja = evaluate_test_set(test_dataset_ja, model, device)
print("ja: ", test_f1_ja, accuracy_ja)

test_f1_es, accuracy_es = evaluate_test_set(test_dataset_es, model, device)
print("es: ", test_f1_es, accuracy_es)

test_f1_zh, accuracy_zh = evaluate_test_set(test_dataset_zh, model, device)
print("zh: ", test_f1_zh, accuracy_zh)
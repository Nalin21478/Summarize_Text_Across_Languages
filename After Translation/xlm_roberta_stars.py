# %%
import json

combined_data_train = []

# Load data from each JSON file, extract 'en' and 'stars', and append them to the combined_data list
files = [
    "/home/pavit21178/misc/Nalin_NLP/translated_train_jsons/de_train_en.json",
    "/home/pavit21178/misc/Nalin_NLP/translated_train_jsons/es_train_en.json",
    "/home/pavit21178/misc/Nalin_NLP/translated_train_jsons/fr_train_en.json",
    "/home/pavit21178/misc/Nalin_NLP/translated_train_jsons/ja_train_en.json",
    "/home/pavit21178/misc/Nalin_NLP/translated_train_jsons/zh_train_en.json"
]

for file_path in files:
    with open(file_path) as f:
        data = json.load(f)
        for item in data:
            combined_data_train.append({'en': item['en'], 'stars': item['stars']})






# %%
import json

combined_data_test = []

# Load data from each JSON file, extract 'en' and 'stars', and append them to the combined_data list
files = [
    "/home/pavit21178/misc/Nalin_NLP/translated_test_jsons/de_test_en.json",
    "/home/pavit21178/misc/Nalin_NLP/translated_test_jsons/es_test_en.json",
    "/home/pavit21178/misc/Nalin_NLP/translated_test_jsons/fr_test_en.json",
    "/home/pavit21178/misc/Nalin_NLP/translated_test_jsons/ja_test_en.json",
    "/home/pavit21178/misc/Nalin_NLP/translated_test_jsons/zh_test_en.json"
]

for file_path in files:
    with open(file_path) as f:
        data = json.load(f)
        for item in data:
            combined_data_test.append({'en': item['en'], 'stars': item['stars']})



# %%
import json

combined_data_val = []

# Load data from each JSON file, extract 'en' and 'stars', and append them to the combined_data list
files = [
    "/home/pavit21178/misc/Nalin_NLP/translate_val_texts/de_val_en.json",
    "/home/pavit21178/misc/Nalin_NLP/translate_val_texts/es_val_en.json",
    "/home/pavit21178/misc/Nalin_NLP/translate_val_texts/fr_val_en.json",
    "/home/pavit21178/misc/Nalin_NLP/translate_val_texts/ja_val_en.json",
    "/home/pavit21178/misc/Nalin_NLP/translate_val_texts/zh_val_en.json"
]

for file_path in files:
    with open(file_path) as f:
        data = json.load(f)
        for item in data:
            combined_data_val.append({'en': item['en'], 'stars': item['stars']})



# %%
def remap_labels(example):
    class_mapping = {1: 0, 2: 1, 4: 2, 5: 3}
    example['stars'] = class_mapping.get(example['stars'])  
    return example

combined_data_train = [remap_labels(example) for example in combined_data_train]
combined_data_test = [remap_labels(example) for example in combined_data_test]
combined_data_val = [remap_labels(example) for example in combined_data_val]



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
combined_data_train[0]

# %%
from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification


tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base')

# %%
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
max_length = 512
max_length = 512
tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base')
device = torch.device('cuda')
model = XLMRobertaForSequenceClassification.from_pretrained('xlm-roberta-base', num_labels=4).to(device)

train_dataset = CustomDataset([item['en'] for item in combined_data_train], [item['stars'] for item in combined_data_train], tokenizer, max_length)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_dataset = CustomDataset([item['en'] for item in combined_data_test], [item['stars'] for item in combined_data_test], tokenizer, max_length)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)
val_dataset = CustomDataset([item['en'] for item in combined_data_val], [item['stars'] for item in combined_data_val], tokenizer, max_length)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=True)



# %%


import torch
from tqdm import tqdm
from sklearn.metrics import f1_score

# Define optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)


train_losses = []
val_losses = []
f1_scores = []

num_epochs = 3
for epoch in range(num_epochs):
    # Training
    model.train()
    train_loss = 0.0
    for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}', unit='batch', leave=False):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
    
    # Compute average training loss for the epoch
    train_loss /= len(train_loader)
    train_losses.append(train_loss)
    
    # Validation
    model.eval()
    val_loss = 0.0
    val_predicted = []
    val_labels = []
    with torch.no_grad():
        for batch in tqdm(val_loader, desc='Validation', unit='batch', leave=False):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            logits = outputs.logits
            
            val_loss += loss.item()  # Accumulate validation loss
            
            _, predicted = torch.max(logits, 1)
            val_predicted.extend(predicted.tolist())
            val_labels.extend(labels.tolist())
            
    # Compute average validation loss for the epoch
    val_loss /= len(val_loader)
    val_losses.append(val_loss)

    # Compute F1 score for the epoch
    f1 = f1_score(val_labels, val_predicted, average='weighted')
    f1_scores.append(f1)

    # Print epoch results
    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss}, Validation Loss: {val_loss}, F1 Score: {f1}")


# Plot training and validation loss curves
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()

# Plot F1 score curve
plt.figure(figsize=(10, 5))
plt.plot(f1_scores, label='F1 Score')
plt.xlabel('Epoch')
plt.ylabel('F1 Score')
plt.title('F1 Score')

plt.legend()
plt.show()


# %%


# %%
model.save_pretrained("translated_xlm-robert_model_stars")

# %%




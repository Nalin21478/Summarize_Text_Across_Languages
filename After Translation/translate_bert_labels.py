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
            combined_data_train.append({'en': item['en'], 'label': item['label']})






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
            combined_data_test.append({'en': item['en'], 'label': item['label']})



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
            combined_data_val.append({'en': item['en'], 'label': item['label']})



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
from transformers import AutoTokenizer, AutoModelForSequenceClassification,AutoConfig


tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# %%
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
max_length = 512

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
device = torch.device('cpu')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2).to(device)

train_dataset = CustomDataset([item['en'] for item in combined_data_train], [item['label'] for item in combined_data_train], tokenizer, max_length)
train_loader = DataLoader(train_dataset, batch_size=8*2, shuffle=True)
test_dataset = CustomDataset([item['en'] for item in combined_data_test], [item['label'] for item in combined_data_test], tokenizer, max_length)
test_loader = DataLoader(test_dataset, batch_size=8*2, shuffle=True)
val_dataset = CustomDataset([item['en'] for item in combined_data_val], [item['label'] for item in combined_data_val], tokenizer, max_length)
val_loader = DataLoader(val_dataset, batch_size=8*2, shuffle=True)



# %%
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import f1_score



import torch
from tqdm import tqdm
from sklearn.metrics import f1_score

# Define optimizer
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import f1_score

# Define training and validation functions
def train(model, optimizer, criterion, train_loader, device):
    model.train()
    total_loss = 0.0
    for batch in tqdm(train_loader, desc='Training', unit='batch', leave=False):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(train_loader)

def evaluate(model, criterion, data_loader, device):
    model.eval()
    total_loss = 0.0
    all_predictions = []
    all_labels = []
    with torch.no_grad():
        for batch in tqdm(data_loader, desc='Evaluation', unit='batch', leave=False):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()
            
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=1)
            all_predictions.extend(predictions.tolist())
            all_labels.extend(labels.tolist())
    
    avg_loss = total_loss / len(data_loader)
    f1 = f1_score(all_labels, all_predictions, average='weighted')
    
    return avg_loss, f1

optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
criterion = torch.nn.CrossEntropyLoss()

# Define device
device = torch.device('cuda')
model.to(device)

# Training loop
num_epochs = 3
for epoch in range(num_epochs):
    train_loss = train(model, optimizer, criterion, train_loader, device)
    val_loss, val_f1 = evaluate(model, criterion, val_loader, device)
    
    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss}, Validation Loss: {val_loss}, Validation F1 Score: {val_f1}")

# %%



model.save_pretrained("bert_translated_model")
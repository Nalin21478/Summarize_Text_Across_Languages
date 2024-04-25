import evaluate
import numpy as np
import json
from transformers import Trainer, TrainingArguments
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import Dataset
import wandb
from huggingface_hub import login
from datasets import load_dataset
import os

id2label = {0: "NEGATIVE", 1: "POSITIVE"}
label2id = {"NEGATIVE": 0, "POSITIVE": 1}

model_id = 'xlm-roberta-base' 
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSequenceClassification.from_pretrained(model_id, num_labels=2, id2label=id2label, label2id=label2id)

os.environ["TOKENIZERS_PARALLELISM"] = "false"
access_token_write = "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"
login(token=access_token_write)
wandb.login(key="cbecd600ce14e66bbbed0c7b4bb7fb317f48a47a", relogin=True)

def get_data(split, k50 = False):
    data = load_dataset("hungnm/multilingual-amazon-review-sentiment-processed", split = split)
    fr_data = data.filter(lambda example: example['language'] == 'de')
    if k50:
        fr_data = fr_data.select(range(50000))
    data_fr = fr_data.map(lambda example: {col: example[col] for col in ['text', 'label']})
    dataset =  data_fr.remove_columns([col for col in data_fr.column_names if col not in ['text', 'label']])

    def tokenize(batch):
        return tokenizer(
            batch["text"], padding="max_length", truncation=True, return_tensors="pt"
        )

    dataset = dataset.map(tokenize, batched=True, remove_columns=["text"])
    return dataset

# train_data = get_data('train', k50 = True)
# val_data = get_data('validation')
test_data = get_data('test')


metric = evaluate.load("f1")
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return metric.compute(predictions=predictions, references=labels)


hugging_id = "greasyFinger/german_xl"

training_args = TrainingArguments(
    output_dir=hugging_id ,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    learning_rate=2e-5,
    num_train_epochs=3,
    torch_compile=True,
    weight_decay=0.01,
    optim="adamw_torch_fused",
    logging_dir=f"{hugging_id}/logs",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    report_to="wandb",
    push_to_hub=True,
    hub_strategy="every_save",
    hub_model_id=hugging_id,
    use_cpu=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    # train_dataset=train_data,
    # eval_dataset=val_data,
    compute_metrics=compute_metrics,
)

# trainer.train()
trainer.evaluate(eval_dataset = test_data)

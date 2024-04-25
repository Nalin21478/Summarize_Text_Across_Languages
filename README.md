# Sentiment Analysis Across Languages: Evaluation Before and After Machine Translation to English

Our project investigates the performance of transformer models in Sentiment Analysis tasks across multilingual datasets and machine-translated text, addressing the lack of sentiment resources for non-English languages. It evaluates model effectiveness in various linguistic contexts, highlighting performance variations and suggesting avenues for future research to improve sentiment analysis across diverse languages.

## Directory Structure

- **Labels wo translation:**
  - Contains scripts for sentiment analysis without translation, focusing on binary label prediction.
  - Two subfolders for BERT and XLM-RoBERTa models.
  - Scripts for fine-tuning and saving models.

- **Stars wo translation:**
  - Contains scripts for sentiment analysis without translation, focusing on star rating prediction.
  - Two subfolders for BERT and XLM-RoBERTa models.
  - Scripts for fine-tuning and saving models.

- **Translation Scripts:**
  - Scripts for translating review sentences from other languages to English.
  - Translated data (train, test, validation) saved in the "Translated Data" directory.
  - Data stored in language-wise (source language) JSON folders.

- **Infer scripts:**
  - Scripts for running inference on test data and obtaining scores for each fine-tuned model.

- **After Translation:**
  - Scripts for fine-tuning English BERT and XLM-RoBERTa models on both tasks using the translated dataset.

## Usage
Can directly run the scripts sequentially to acquire model checkpoints and eventaully infer and compare models.
To directly infer, models import specific model checkpoint either from hugging face using the provided access key or load check point from 
[drive link](https://drive.google.com/drive/folders/1R00drvoxtaUIxxDaJxbdy2yfBKd2mkQ5)

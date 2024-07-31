#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 07:00:14 2024

@author: msayed
"""

import pandas as pd
from sklearn.model_selection import train_test_split
import time
from datasets import load_dataset, DatasetDict, Dataset

from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer)

from peft import PeftModel, PeftConfig, get_peft_model, LoraConfig
import evaluate
import torch
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef



import pandas as pd

startTime = time.time() 

# To load file locally
data_url = 'dataset.csv'

# To load file form mendaly
#data_url = 'https://data.mendeley.com/datasets/c4n7fckkz3/1/files/e7d37892-8b22-4bd2-a5cd-854cc155e4c1'



df = pd.read_csv(data_url, on_bad_lines='skip')

df.columns = ['user_ip', 'domain', 'timestamp', 'attack', 'request', 'len', 'subdomains_count', 'w_count',\
              'w_max', 'entropy', 'w_max_ratio', 'w_count_ratio', 'digits_ratio', 'uppercase_ratio', \
              'time_avg', 'time_stdev', 'size_avg', 'size_stdev', 'throughput', 'unique', 'entropy_avg', \
              'entropy_stdev']



df = df[['attack', 'request']]

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')


# Display the first few rows of the DataFrame
print('DataFrame shape :', df.shape)
print('First few rows of data : \n', df.head())



num_data = int(input("Enter a number of data you want to consider for Attack and Not Attack type domain(individually) : "))

balanced_df = df.groupby('attack').sample(n=num_data, random_state=1).reset_index(drop=True)

#balanced_df['Type'] = balanced_df['Type'].str.replace('DGA', 1)
#balanced_df['Type'] = balanced_df['Type'].str.replace('Normal', 0)

#balanced_df['attack'] = balanced_df['attack'].replace({'True': 1, 'False': 0})
balanced_df['attack'] = balanced_df['attack'].astype(int)

#print(balanced_df)

# Display the first few rows of the balanced DataFrame
print('Balanced DataFrame shape :', balanced_df.shape)
print('First few rows of data : \n', balanced_df.head())


# Shuffle the DataFrame
balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)

balanced_df.rename(columns={'attack': 'label'}, inplace=True)

print('Now spliting dataset as training (60%), validation(20%), and test(20%) and creating datasetdict')
# Separate the data based on type
df_type1 = balanced_df[balanced_df['label'] == 1]
df_type2 = balanced_df[balanced_df['label'] == 0]

# Split type1 data
type1_train, type1_temp = train_test_split(df_type1, test_size=0.4, random_state=42)
type1_val, type1_test = train_test_split(type1_temp, test_size=0.5, random_state=42)

# Split type2 data
type2_train, type2_temp = train_test_split(df_type2, test_size=0.4, random_state=42)
type2_val, type2_test = train_test_split(type2_temp, test_size=0.5, random_state=42)

# Combine type1 and type2 data for each set
train_df = pd.concat([type1_train, type2_train]).sample(frac=1, random_state=42).reset_index(drop=True)
val_df = pd.concat([type1_val, type2_val]).sample(frac=1, random_state=42).reset_index(drop=True)
test_df = pd.concat([type1_test, type2_test]).sample(frac=1, random_state=42).reset_index(drop=True)



print('train : ', train_df.head())
print('val : ', val_df.head())
print('test : ', test_df.head())

dataset = DatasetDict({'train':Dataset.from_dict(train_df.to_dict(orient='list')),\
            'val':Dataset.from_dict(val_df.to_dict(orient='list')),\
            'test':Dataset.from_dict(test_df.to_dict(orient='list'))})
print('Dataset :', dataset)


model_name = 'distilbert-base-uncased'
# model_checkpoint = 'roberta-base' # you can alternatively use roberta-base but this model is bigger thus training will take longer

# define label maps
id2label = {0: 'No', 1: 'Yes'}
label2id = {'No': 0, 'Yes': 1}

# generate classification model from model_checkpoint
model = AutoModelForSequenceClassification.from_pretrained(
    model_name, num_labels=2, id2label=id2label, label2id=label2id)

print('Model : ', model)

tokenizer = AutoTokenizer.from_pretrained(model_name, add_prefix_space=True)

# add pad token if none exists
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(tokenizer))

# create tokenize function
def tokenize_function(examples):
    # extract text
    text = examples["request"]

    #tokenize and truncate text
    tokenizer.truncation_side = "left"
    tokenized_inputs = tokenizer(
        text,
        return_tensors="np",
        truncation=True,
        max_length=512
    )
    return tokenized_inputs
    
    

tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=['request'])
print("Tokenized Dataset : ", tokenized_dataset)


# create data collator
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

accuracy = evaluate.load("accuracy")

# define an evaluation function to pass into trainer later
def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=1)

    return {"accuracy": accuracy.compute(predictions=predictions, references=labels)}

peft_config = LoraConfig(task_type="SEQ_CLS",
                        r=4,
                        lora_alpha=32,
                        lora_dropout=0.01,
                        target_modules = ['q_lin'])
print('Peft Config : ', peft_config)

model = get_peft_model(model, peft_config)
print('Peft model description : ', model.print_trainable_parameters())

# hyperparameters
lr = 1e-3
batch_size = 4
num_epochs = 10

# define training arguments
training_args = TrainingArguments(
    output_dir= model_name + "-lora-text-classification",
    learning_rate=lr,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=num_epochs,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
)

# creater trainer object
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["val"],
    tokenizer=tokenizer,
    data_collator=data_collator, # this will dynamically pad examples in each batch to be equal length
    compute_metrics=compute_metrics,
)

# train model
trainer.train()

# Evaluate model on the test dataset
predictions = trainer.predict(tokenized_dataset["test"])

# Compute metrics
preds = np.argmax(predictions.predictions, axis=-1)

# Confusion Matrix
tn, fp, fn, tp = confusion_matrix(tokenized_dataset["test"]['label'], preds).ravel()

# Accuracy
accurac = (tp + tn) / float(tp + tn + fp + fn)

# Precision (Positive Predictive Value)
precision = tp / float(tp + fp)

# Recall (Sensitivity or True Positive Rate)
recall = tp / float(tp + fn)

# F1 Score
f1 = 2 * (precision * recall) / (precision + recall)

# Matthews Correlation Coefficient
mcc = matthews_corrcoef(tokenized_dataset["test"]['label'], preds)

# Specificity (True Negative Rate)
specificity = tn / (tn + fp)

# Negative Predictive Value
npv = tn / (tn + fn) if (tn + fn) != 0 else 0

# False Positive Rate
fpr = fp / (fp + tn)

# False Discovery Rate
fdr = fp / (fp + tp) if (fp + tp) != 0 else 0

# False Negative Rate
fnr = fn / (fn + tp)

print('Finetuned lora based distilbert model performance on different matrics\n')
print("Sensitivity (Recall):", recall)
print("Specificity:", specificity)
print("Precision (Positive Predictive Value):", precision)
print("Negative Predictive Value:", npv)
print("False Positive Rate:", fpr)
print("False Discovery Rate:", fdr)
print("False Negative Rate:", fnr)
print("Accuracy:", accurac)
print("F1 Score:", f1)
print("Matthews Correlation Coefficient:", mcc)




print("Total time taken in seconds = ", str(time.time() - startTime))

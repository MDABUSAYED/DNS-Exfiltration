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
from sklearn.preprocessing import MinMaxScaler


import pandas as pd

import torch
import torch.nn as nn
from transformers import BertModel
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset


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

    
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')


# Display the first few rows of the DataFrame
print('DataFrame shape :', df.shape)
print('First few rows of data : \n', df.head())

    
# filter request, point 2 & 3
df['request'] = df['request'].str.replace('.dnsresearch.ml', '')
contains_str = df['request'].str.contains('DLJ5PGNNBZAGTKWO6ECYJBP427WZFA')
df = df[~contains_str]

# point 1
contains_domain1 = df['domain'].str.contains('e5.sk')
balanced_df1 = df[contains_domain1].groupby('attack').sample(n=1000, random_state=1).reset_index(drop=True)

contains_domain2 = df['domain'].str.contains('mcafee')
balanced_df2 = df[contains_domain2].groupby('attack').sample(n=1000, random_state=1).reset_index(drop=True)


balanced_df3 = df[~contains_domain1 & ~contains_domain2].groupby('attack').sample(n=3000, random_state=1).reset_index(drop=True)

balanced_df = pd.concat([balanced_df1, balanced_df2, balanced_df3], ignore_index=True)


#balanced_df = balanced_df[['attack', 'request']]

continuous_features = ['len', 'subdomains_count', 'w_count','w_max', 'entropy', 'w_max_ratio',\
                       'w_count_ratio', 'digits_ratio', 'uppercase_ratio']

balanced_df[continuous_features] = MinMaxScaler().fit_transform(balanced_df[continuous_features])




#num_data = int(input("Enter a number of data you want to consider for Attack and Not Attack type domain(individually) : "))

#balanced_df = df.groupby('attack').sample(n=num_data, random_state=1).reset_index(drop=True)

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
        padding='max_length',
        max_length=128
    )
    return tokenized_inputs
    

tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=['user_ip', 'domain','request', 'timestamp','time_avg', 'time_stdev', 'size_avg', 'size_stdev', 'throughput', 'unique', 'entropy_avg', 'entropy_stdev'])

print('tokenized_dataset ', tokenized_dataset)


def calculate_accuracy(outputs, labels):
    preds = torch.argmax(outputs, dim=1) #torch.round(torch.sigmoid(outputs))
    #print(preds, labels)
    correct = (preds == labels).sum().item()
    accuracy = correct / labels.size(0)
    return accuracy

class BertWithNumeric(nn.Module):
    def __init__(self, bert_model_name, num_numeric_features):
        super(BertWithNumeric, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.numeric_embedding = nn.Linear(num_numeric_features, 768)
        self.classifier = nn.Linear(768 * 2, 2)

    def forward(self, input_ids, attention_mask, numeric_features):
        bert_outputs = self.bert(input_ids=input_ids,
                                 attention_mask=attention_mask)
        text_embeddings = bert_outputs.last_hidden_state[:, 0, :]
        numeric_embeddings = self.numeric_embedding(numeric_features)
        combined = torch.cat((text_embeddings, numeric_embeddings), dim=1)
        logits = self.classifier(combined)
        return logits

# Example usage with dummy data
model = BertWithNumeric(model_name, num_numeric_features=9)
optimizer = Adam(model.parameters(), lr=1e-5)
criterion = nn.CrossEntropyLoss()

# Dummy data for illustration
input_ids = torch.tensor(tokenized_dataset['train']['input_ids'])
attention_mask = torch.tensor(tokenized_dataset['train']['attention_mask'])
numeric_features = torch.transpose(torch.tensor([tokenized_dataset['train'][f] for f in continuous_features]), 0, 1)
labels = torch.tensor(tokenized_dataset['train']['label'])

# Create DataLoader
dataset = TensorDataset(input_ids, attention_mask, numeric_features, labels)
dataloader = DataLoader(dataset, batch_size=16)

# Training loop
model.train()
for epoch in range(5):  # Number of epochs
    total_train_loss = 0
    total_train_acc = 0
    model.train()
    for batch in dataloader:
        batch_input_ids, batch_attention_mask, batch_numeric_features, batch_labels = batch

        optimizer.zero_grad()
        outputs = model(batch_input_ids, batch_attention_mask, batch_numeric_features)
        loss = criterion(outputs, batch_labels)
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()
        #print(outputs, batch_labels)
        total_train_acc += calculate_accuracy(outputs, batch_labels)
        #print(f'Epoch {epoch + 1}, Loss: {loss.item()}')

    avg_train_loss = total_train_loss / len(dataloader)
    avg_train_acc = total_train_acc / len(dataloader)
    
    print(f'Epoch {epoch + 1}\tTraining Loss: {avg_train_loss:.6f}\tAccuracy: {{"accuracy": {avg_train_acc:.3f}}}')
    # Inference (predicting)
    model.eval()
    total_val_loss = 0
    total_val_acc = 0
    with torch.no_grad():
        outputs = model(torch.tensor(tokenized_dataset['val']['input_ids']), torch.tensor(tokenized_dataset['val']['attention_mask']), torch.transpose(torch.tensor([tokenized_dataset['val'][f] for f in continuous_features]), 0, 1))
        labels = torch.tensor(tokenized_dataset['val']['label'])
        predictions = torch.argmax(outputs, dim=1)
        print(f'Predictions: {predictions}')
        total_val_loss += loss.item()
        total_val_acc += calculate_accuracy(outputs, labels)

    print(f'val Loss: {total_val_loss:.6f}\tval Accuracy: {{"accuracy": {total_val_acc:.3f}}}')

# Inference (predicting)
model.eval()
total_test_loss = 0
total_test_acc = 0
with torch.no_grad():
    outputs = model(torch.tensor(tokenized_dataset['test']['input_ids']), torch.tensor(tokenized_dataset['test']['attention_mask']), torch.transpose(torch.tensor([tokenized_dataset['test'][f] for f in continuous_features]), 0, 1))
    labels = torch.tensor(tokenized_dataset['test']['label'])
    predictions = torch.argmax(outputs, dim=1)
    print(f'Predictions: {predictions}')
    total_test_loss += loss.item()
    total_test_acc += calculate_accuracy(outputs, labels)

print(f'Test Loss: {total_test_loss:.6f}\tTest Accuracy: {{"accuracy": {total_test_acc:.3f}}}')




print("Total time taken in seconds = ", str(time.time() - startTime))










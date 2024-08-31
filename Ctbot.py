import numpy as np          
import nltk
import json
import flask
import pandas as pd
import torch 


cls=[]
text=[]
labels=[]
ignore=["?","!",",","'s"]

data_file = open('intent.json').read()
data=json.loads(data_file)


from transformers import BertTokenizer,BertModel # type: ignore


# Load the BERT model (TensorFlow)
model = BertModel.from_pretrained('bert-base-uncased')


for intent in data['intents']:
    for pattern in intent['examples']:
        text.append('examples')
        labels.append('Intent')

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
inputs = tokenizer(text.tolist(), return_tensors='pt', max_length=512, truncation=True, padding='max_length')

from sklearn.preprocessing import LabelEncoder # type: ignore
label_encoder=LabelEncoder()
label_encoder.fit_transform(labels)

from sklearn.model_selection import train_test_split
train_inputs,test_inputs,train_labels,test_labels=train_test_split(inputs['input_ids'],label_encoder,test_size=0.2,random_state=0)
from transformers import BertForSequenceClassification, Trainer, TrainingArguments

model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(set(label_encoder)))

training_arg=TrainingArguments(
     output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
)
trainer = Trainer(
    model=model,
    args=training_arg,
    train_dataset=train_inputs,
    eval_dataset=test_inputs,
)
trainer.train()
def predict_intent(text):

    # Tokenize the input text
    inputs = tokenizer(text, return_tensors='pt', max_length=128, truncation=True, padding=True)
    
    # Predict the intent
    outputs = model(**inputs)
    prediction = outputs.logits.argmax(dim=-1).item()
    
    # Convert the numeric prediction back to the original intent
    predicted_intent = label_encoder.inverse_transform([prediction])[0]
    return predicted_intent
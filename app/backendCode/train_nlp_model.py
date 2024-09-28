import json
import numpy as np 
from nltk.stem import WordNetLemmatizer
import torch 
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from nlp_model import tokenize,bag_of_words
import nltk
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()
with open('Intent.json','r') as file:
    data = json.load(file)
    
all_words = []
tags = []
xy = []

for intent in data['Intents']:
    tag = intent['tag']
    tags.append(tag)
    for pattern in intent['examples']:
        W = tokenize(pattern)
        all_words.extend(W)
        xy.append((W,tag))

ignore = ['?','!','.',',','@']

all_words = [lemmatizer.lemmatize(W) for W in all_words if W not in ignore]
all_words = sorted(set(all_words))
tags = sorted(set(tags))


X_train = []
Y_train = []

for (pattern_sentence, tag) in xy:
    bag = bag_of_words(pattern_sentence,all_words)
    X_train.append(bag)
    
    label = tags.index(tag)
    Y_train.append(label)

X_train = np.array(X_train)
Y_train =np.array(Y_train)

class ChatDataset(Dataset):
    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.Y_data = Y_train
     
    #dataset[idx]   
    def __getitem__(self, index):
        return self.x_data[index], self.Y_data[index]
    
    def __len__(self):
        return self.n_samples
    
batch_size = 8
hidden_layer = 8
output_size = len(tags)
input_size = len(all_words)
learning_rate = 0.01
num_epochs = 500


dataset = ChatDataset()
train_loader = DataLoader(dataset = dataset, batch_size = batch_size,shuffle=True,num_workers=0)

from nlop import NeuralNet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = NeuralNet(input_size, hidden_layer,output_size)           


criterion = nn. CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)

for epoch in range (num_epochs):
    for (words, label) in train_loader:
        words = words.to(device).float()
        labels =label.to(device)
        
        output = model(words)
        loss = criterion(output,labels.long())
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    if(epoch + 1) % 100 == 0:
        print(f'epoch{epoch+1}/{num_epochs},loss = {loss.item():.4f}')

print(f'final loss , loss = {loss.item():.4f}')

data = {
    "model_state" : model.state_dict(),
    "input_size" : input_size,
    "output_size": output_size,
    "hidden_layer" : hidden_layer,
    "all_words" : all_words,
    "tags" : tags
    }

FILE = "result.pth"
torch.save(data,FILE)

print(f'training complete. file saved to {FILE}')

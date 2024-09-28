import random
import json
import torch
from feedforward_net import NeuralNet
from nlp_model import tokenize, bag_of_words
from nrclex  import  NRCLex
from datetime  import datetime

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load intents and model data
with open('Intent.json', 'r') as file:
    intents = json.load(file)

FILE = 'result.pth'
data = torch.load(FILE)

input_size = data['input_size']
hidden_layer = data['hidden_layer']
output_size = data['output_size']
all_words = data['all_words']
tags = data['tags']
model_state = data['model_state']

model = NeuralNet(input_size, hidden_layer, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "DhiTvam"
print("Let's chat! : :)")
print("Type 'q' to exit")

def detect_emotion(text):
    emotion = NRCLex(text)
    emotions = emotion.raw_emotion_scores
    return emotions

def responses_based_on_emotion(emotions):
    if emotions.get('anger',0) > 0.2:
        return "I see you're feeling angry. How can I help you"
    elif emotions.get('sadness',0)>0.15:
        return "It seems like you're feeling sad. Do you want to talk?"
    elif emotions.get('fear',0)>0.1:
        return "You seem worried . I'm here to help."
   
def get_response(tag) :
    if tag == "datetime":
        now = datetime.now()
        current_time = now.strftime("%d-%m-%Y %H:%M:%S")
        #print(current_time)
        return f"{bot_name} : Sure! The current Date and time is {current_time}"
    else:
        for intent in intents["Intents"]:
        
            if intent["tag"] == tag:
                return f"{bot_name}: {random.choice(intent['responses'])}"
           
        return (f"{bot_name}: Could you say that in another way.........")

while True:
    sentence = input("you: ").lower()  # Convert input to lowercase for case-insensitive matching
    
    if sentence in ['q', 'quit']:
        print(f"{bot_name}: Bye! Take care")
        break
    
    emotions = detect_emotion(sentence)
    emotion_response = responses_based_on_emotion(emotions)
    
    if emotion_response:  
        print(f"{bot_name}: {emotion_response}")
        continue
    
    
    # Process user input and get model prediction
    sentence = tokenize(sentence)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).float().to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]

   # print(f"Predicted tag: {tag}")
    
    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]

    # Check the probability and respond
    if prob.item() > 0.8:
        response = get_response(tag)
        print(response)
    else:
        print(f"{bot_name}: Could you say that in another way.........")

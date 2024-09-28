from flask import Flask, render_template, redirect, url_for, request, flash, jsonify, session
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
import os
import json
import torch
import random
from nlop import NeuralNet
from model_training import tokenize, bag_of_words
from nrclex import NRCLex
from datetime import datetime

app = Flask(__name__, template_folder='C:\\Users\\kavana s\\OneDrive\\Desktop\\Chatbhot\\templates')
app.config['SECRET_KEY'] = os.urandom(24)
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+mysqlconnector://root:Anishakaaval35@localhost/chatbot_db'
db = SQLAlchemy(app)
bcrypt = Bcrypt(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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

# Database model for User
class User(db.Model, UserMixin):
    __tablename__ = 'users'
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), unique=True, nullable=False)
    email = db.Column(db.String(100), unique=True, nullable=False)
    password = db.Column(db.String(255), nullable=False)
    created_at = db.Column(db.DateTime, default=db.func.current_timestamp())

class Chat(db.Model):
    __tablename__ = 'chats'
    chat_id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    message = db.Column(db.Text, nullable=False)
    bot_response = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, default=db.func.current_timestamp())
    
    user = db.relationship('User', backref=db.backref('chats', lazy=True))


with app.app_context():
    db.create_all()


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Emotion detection using NRCLex
def detect_emotion(text):
    emotion = NRCLex(text)
    return emotion.raw_emotion_scores

# Custom chatbot response based on emotions
def responses_based_on_emotion(emotions):
    if emotions.get('anger', 0) > 0.2:
        return "I see you're feeling angry. How can I help you?"
    elif emotions.get('sadness', 0) > 0.15:
        return "It seems like you're feeling sad. Do you want to talk?"
    elif emotions.get('fear', 0) > 0.1:
        return "You seem worried. I'm here to help."

def get_response(tag):
    if tag == "datetime":
        now = datetime.now()
        current_time = now.strftime("%d-%m-%Y %H:%M:%S")
        return f": Sure! The current date and time is {current_time}"
    else:
        for intent in intents["Intents"]:
            if intent["tag"] == tag:
                return f": {random.choice(intent['responses'])}"
        return f": Could you say that in another way?"



@app.route('/')
def home():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        # Check if form data is being received correctly
        print(f"Username entered: {username}")
        print(f"Password entered: {password}")
        
        user = User.query.filter_by(username=username).first()
        
        if user:
            print(f"User {username} found in the database.")
        else:
            print("User not found.")

        if user and bcrypt.check_password_hash(user.password, password):
            print("Password matched.")
            session['username'] = username  
            login_user(user)
            flash('Login successful!', 'success')
            return redirect(url_for('web'))  
        else:
            print("Password did not match.")
            flash('Login Unsuccessful. Please check your username and password.', 'danger')
            return redirect(url_for('login'))  

    return render_template('login.html')

@app.route('/test_redirect')
def test_redirect():
    return redirect(url_for('web'))

@app.route('/dashboard')
def dashboard():
    if 'username' in session:
        return f"Welcome to the dashboard, {session['username']}!"
    else:
        return redirect(url_for('login'))

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')

        existing_user = User.query.filter_by(username=username).first()
        existing_email = User.query.filter_by(email=email).first()

        if existing_user:
            flash('Username already exists. Please choose a different one.', 'danger')
            return redirect(url_for('signup'))
        
        if existing_email:
            flash('Email already registered. Please use a different email.', 'danger')
            return redirect(url_for('signup'))

        hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')
        new_user = User(username=username, email=email, password=hashed_password)
        db.session.add(new_user)
        db.session.commit()

        flash(f'Account created for {username}!', 'success')
        return redirect(url_for('web'))
    return render_template('signup.html')

@app.route('/get_response', methods=['POST'])
def chatbot_response():
    user_input = request.form['message']
    
    
    emotions = detect_emotion(user_input)
    emotion_response = responses_based_on_emotion(emotions)
    
    if emotion_response:
        return jsonify({"response": emotion_response})
    
   
    sentence = tokenize(user_input)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).float().to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]

    if prob.item() > 0.8:
        response = get_response(tag)
        return jsonify({"response": response})
    else:
        return jsonify({"response": f" : Could you say that in another way?"})

@app.route('/web')
def web():
    return render_template('web.html')

@app.route('/logout')
@login_required
def logout():
    logout_user() 
    flash('You have been logged out!', 'success')  
    return redirect(url_for('login'))


if __name__ == '__main__':
    app.run(debug=True)

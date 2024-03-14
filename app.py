from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.tree import DecisionTreeClassifier
from googletrans import Translator

app = Flask(__name__)

# Load the trained model and CountVectorizer
data = pd.read_csv("./train.csv/train.csv")
data["labels"] = data["class"].map({0: "Hate Speech", 1: "Offensive Speech", 2: "No Hate and Offensive Speech"})
data = data[["tweet", "labels"]]

cv = CountVectorizer()
X = cv.fit_transform(data["tweet"])
y = data["labels"]
model = DecisionTreeClassifier()
model.fit(X, y)

# Function to translate text
def translate_text(text, target_language='en'):
    translator = Translator()
    translated_text = translator.translate(text, dest=target_language)
    return translated_text.text

# Function to clean input text
def clean(text):
    # Your cleaning code here
    return text

# Route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Route for handling form submission
@app.route('/check', methods=['POST'])
def check():
    text = request.form['text']
    cleaned_text = clean(text)
    translated_text = translate_text(cleaned_text)
    input_vector = cv.transform([translated_text])
    prediction = model.predict(input_vector)[0]
    return render_template('result.html', text=text, prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)

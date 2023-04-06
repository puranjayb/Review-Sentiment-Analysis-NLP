import pickle
import os

from flask import Flask, request, render_template

from processing import preprocessing
from sklearn.feature_extraction.text import CountVectorizer

app = Flask(__name__)
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    review = request.form['review']
    processed_review = preprocessing(review)
    input_data = vectorizer.transform([' '.join(processed_review)])
    prediction = model.predict(input_data)
    if prediction[0] == 1:
        sentiment = 'Positive'
    else:
        sentiment = 'Negative'
    return render_template('index.html', prediction=sentiment)

if __name__ == '__main__':
    app.run(debug=True)

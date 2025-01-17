from flask import Flask, request, jsonify, render_template
import xgboost as xgb
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Initialize the stemmer
stemmer = SnowballStemmer(language='english')

# Tokenizer function
def tokenize(text):
    return [stemmer.stem(token) for token in word_tokenize(text) if token.isalpha()]

# Load vectorizer vocabulary and initialize the vectorizer
with open('vectorizer_vocabulary.json', 'r') as f:
    vocabulary = json.load(f)

vectorizer = TfidfVectorizer(
    tokenizer=tokenize,
    vocabulary=vocabulary,
    ngram_range=(1, 2),
    max_features=1000
)

# Load the trained XGBoost model
xgb_model = xgb.Booster()
xgb_model.load_model('xgb_model.json')  # Load the model saved in XGBoost's native format

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the comment from the form
        comment = request.form.get('comment', '')
        
        if not comment.strip():
            return jsonify({'error': 'Please provide a valid comment.'}), 400

        # Preprocess and transform the comment
        preprocessed_comment = tokenize(comment)
        comment_list = [' '.join(preprocessed_comment)]  # Convert back to string for vectorizer
        comment_vector = vectorizer.transform(comment_list)

        # Prepare the input for XGBoost
        dmatrix_input = xgb.DMatrix(comment_vector.toarray())
        
        # Make prediction
        prediction = xgb_model.predict(dmatrix_input)[0]
        
        # Map the prediction to sentiment
        sentiment = "Positive" if prediction >= 0.5 else "Negative"
        
        return jsonify({
            'comment': comment,
            'sentiment': sentiment
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)

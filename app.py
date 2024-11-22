from flask import Flask, redirect, request, render_template
import numpy as np
import pickle
import warnings
import nltk
from features import FeatureExtraction  # Feature extraction for URL phishing detection
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import string

# Initialize Flask app
app = Flask(__name__)

warnings.filterwarnings('ignore')

# Load the models
url_classifier_model = pickle.load(open("pickle/model.pkl", "rb"))  # URL phishing classifier
spam_classifier_model = pickle.load(open('pickle/spam_model.pkl', 'rb'))  # Spam classifier
tfidf_vectorizer = pickle.load(open('pickle/vectorizer.pkl', 'rb'))  # Vectorizer for spam

# NLTK initialization for spam classifier
ps = PorterStemmer()
nltk.download('punkt')
nltk.download('stopwords')

# Helper function to preprocess text for spam classification
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

# Main page route
@app.route("/", methods=["GET", "POST"])
def index():
    return render_template("index.html")

# Route for URL classification
@app.route("/url_classify", methods=["POST"])
def url_classify():
    if request.method == "POST":
        url = request.form["url"]

        # Extract features from the URL
        obj = FeatureExtraction(url)
        x = np.array(obj.getFeaturesList()).reshape(1, 30)

        # Predict whether it's phishing or not
        y_pred = url_classifier_model.predict(x)[0]
        y_pro_phishing = url_classifier_model.predict_proba(x)[0, 0]  # Probability of being phishing
        y_pro_non_phishing = url_classifier_model.predict_proba(x)[0, 1]  # Probability of being safe

        if y_pred == 1:
            pred = "It is {0:.2f}% safe to visit this website.".format(y_pro_non_phishing * 100)
        else:
            pred = "Warning: This website has a {0:.2f}% chance of being phishing.".format(y_pro_phishing * 100)

        return render_template('index.html', url_result=pred, active_tab="url")

# Route for spam classification
@app.route("/spam_classify", methods=["POST"])
def spam_classify():
    if request.method == "POST":
        input_sms = request.form["message"]
        transformed_sms = transform_text(input_sms)
        vector_input = tfidf_vectorizer.transform([transformed_sms])

        result = spam_classifier_model.predict(vector_input)[0]

        if result == 1:
            spam_result = "Spam"
        else:
            spam_result = "Not Spam"

        return render_template('index.html', spam_result=spam_result, active_tab="sms")

# Route for About Page
@app.route('/about')
def about():
    return render_template('about.html')

# Route for Login Page
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        # Add your login validation logic here
        return redirect('/')
    return render_template('login.html')

# Run Flask app
if __name__ == "__main__":
    app.run(debug=True)

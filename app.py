from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pickle
from features import FeatureExtraction  # Feature extraction for URL phishing detection

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the model
url_classifier_model = pickle.load(open("pickle/model.pkl", "rb"))

@app.route("/url_classify", methods=["POST"])
def url_classify():
    try:
        data = request.json
        url = data.get("url", "")
        obj = FeatureExtraction(url)
        x = np.array(obj.getFeaturesList()).reshape(1, 30)

        y_pred = url_classifier_model.predict(x)[0]
        y_pro_phishing = url_classifier_model.predict_proba(x)[0, 0]  # Probability of being phishing
        y_pro_non_phishing = url_classifier_model.predict_proba(x)[0, 1]  # Probability of being safe

        if y_pred == 1:
            return jsonify({"status": "safe", "message": f"{y_pro_non_phishing * 100:.2f}% safe to visit."})
        else:
            return jsonify({"status": "phishing", "message": f"{y_pro_phishing * 100:.2f}% chance of being phishing."})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)

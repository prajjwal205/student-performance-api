from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import os

app = Flask(__name__)
CORS(app)

model = joblib.load("student_model.pkl")
columns = joblib.load("model_columns.pkl")

@app.route("/", methods=["GET"])
def home():
    return "Student Performance Prediction API is running!"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json

    df_input = pd.DataFrame([data])

    df_encoded = pd.get_dummies(df_input, drop_first=True)
    df_encoded = df_encoded.reindex(columns=columns, fill_value=0)

    prediction = model.predict(df_encoded)[0]

    return jsonify({"predicted_percentage": float(prediction)})
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

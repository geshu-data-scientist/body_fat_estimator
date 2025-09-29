from flask import Flask, request, render_template
import joblib
import pandas as pd
import pickle

app = Flask(__name__)

# Load trained model
model = joblib.load("bodyfat_rf_model.pkl")

# Load expected feature columns
with open("bodyfat_features.pkl", "rb") as f:
    feature_columns = pickle.load(f)

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    if request.method == "POST":
        # Collect input values dynamically
        input_data = []
        for feature in feature_columns:
            value = float(request.form[feature])
            input_data.append(value)

        # Prepare dataframe in the correct feature order
        input_df = pd.DataFrame([input_data], columns=feature_columns)

        # Predict
        prediction = model.predict(input_df)[0]

    return render_template("index.html", features=feature_columns, prediction=prediction)

if __name__ == "__main__":
    app.run(debug=False)

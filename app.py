from flask import Flask, request, jsonify
import pandas as pd
import pickle

app = Flask(__name__)

# Load the trained LightGBM model
model_filename = "model_lgbm_optimized.pkl"
with open(model_filename, "rb") as file:
    model = pickle.load(file)


@app.route("/", methods=["GET"])
def home():
    return "Welcome to the Cardiovascular Disease Prediction API!"


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Extracting form data
        data_dict = {key: float(value) for key, value in request.form.items()}
        query_df = pd.DataFrame([data_dict])

        # Make prediction
        prediction = model.predict(query_df)

        # Ensure output is in a serializable format
        prediction_list = prediction.tolist()
        return jsonify({"prediction": prediction_list})
    except Exception as e:
        return jsonify({"error": str(e)})


if __name__ == "__main__":
    app.run(host='0.0.0.0')

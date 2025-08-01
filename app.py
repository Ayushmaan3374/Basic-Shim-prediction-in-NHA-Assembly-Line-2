from flask import Flask, request, jsonify
import pandas as pd
import lightgbm as lgb
import joblib
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load the trained LightGBM model
lgbmr_model = joblib.load("lgbm.pkl")

@app.route("/", methods=["GET"])
def home():
    return "{Working}"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get the TYPE value from the request
        prediction_type = request.form.get("TYPE")

        if prediction_type is None:
            return jsonify({"error": "TYPE missing"}), 400

        # Load both CSV files
        df = pd.read_csv("M_dataset.csv")
        whole_df = pd.read_csv("FTB_Data_2025_06_19.csv")

        # Ensure necessary columns are present
        required_columns = ["LHSSHIM", "RHSSHIM", "TYPE"]
        for col in required_columns:
            if col not in df.columns:
                return jsonify({"error": f"Missing column: {col}"}), 400

        # Filter the main dataset for matching TYPE
        filtered_df = df[df["TYPE"] == prediction_type]
        if filtered_df.empty:
            return jsonify({"error": f"No rows with TYPE = '{prediction_type}' found"}), 400

        # Prepare features for prediction
        selected_features = filtered_df[["LHSSHIM", "RHSSHIM", "TYPE"]].copy()
        selected_features["TYPE"] = selected_features["TYPE"].astype("category")

        # Make prediction using the loaded model
        prediction = lgbmr_model.predict(selected_features)

        # Filter and get the last 30 BASICSHIM values for the same TYPE
        filtered_whole_df = whole_df[whole_df["TYPE"] == prediction_type]
        past_30_values = (
            filtered_whole_df[["BASICSHIM", "LHSSHIM", "RHSSHIM"]]
            .head(30)
            .to_dict(orient="records")
        )

        # Return the prediction and past values
        return jsonify({
            "prediction": prediction.tolist(),
            "type_received": prediction_type,
            "Past30_values": past_30_values
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)

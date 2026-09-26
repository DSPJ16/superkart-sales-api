import joblib
import pandas as pd
from flask import Flask, request, jsonify

# Create the Flask app
app = Flask("SuperKart Sales Predictor")

# Load the trained model once at startup
model = joblib.load("superkart_sales_model_v1_0.joblib")

# Feature order the model was trained on
feature_order = [
    "Product_Weight",
    "Product_Sugar_Content",
    "Product_Allocated_Area",
    "Product_MRP",
    "Store_Size",
    "Store_Location_City_Type",
    "Store_Type",
    "Product_Id_char",
    "Store_Age_Years",
    "Product_Type_Category"
]


# Health check endpoint
@app.get("/")
def home():
    return "SuperKart Sales Prediction API is running"


# Endpoint for single prediction
@app.post("/v1/sales")
def predict_single():
    # Get the JSON data sent in the request
    data = request.get_json()

    # Build a dictionary in the correct feature order
    row = {}
    for feature in feature_order:
        row[feature] = data[feature]

    # Convert to a DataFrame with one row
    input_df = pd.DataFrame([row])

    # Predict
    prediction = model.predict(input_df)[0]
    prediction = float(prediction)

    # Return the prediction as JSON
    return jsonify({"Predicted_Sales": round(prediction, 2)})


# Endpoint for batch prediction
@app.post("/v1/salesbatch")
def predict_batch():
    # Get the uploaded CSV file
    file = request.files["file"]

    # Read it into a DataFrame
    input_df = pd.read_csv(file)

    # Make predictions on the whole file
    predictions = model.predict(input_df[feature_order])

    # Add the predictions as a new column
    input_df["Predicted_Sales"] = predictions.round(2)

    # Return the results as JSON
    return jsonify(input_df.to_dict(orient="records"))


# Run the app locally
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860, debug=False)

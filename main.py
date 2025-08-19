from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import numpy as np
import pickle
import os

# Paths & Model Loading
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_FILE = os.path.join(BASE_DIR, "linear_regression_model.pkl")
SCALER_FILE = os.path.join(BASE_DIR, "scaler.pkl")

try:
    with open(MODEL_FILE, "rb") as file:
        model = pickle.load(file)
    with open(SCALER_FILE, "rb") as file:
        scaler = pickle.load(file)
    print(" Model and scaler loaded successfully.")
except FileNotFoundError as e:
    raise RuntimeError(f" Required file missing: {e}")

# Development Backend through FASTAPI For Production ready 
app = FastAPI(
    title="House Price Prediction API",
    description="Predict house prices using a trained Linear Regression model.",
    version="1.0.0"
)


# Input Schema
class HouseFeatures(BaseModel):
    MedInc: float
    HouseAge: float
    AveRooms: float
    AveBedrms: float
    AvePop: float

# Output Schema
class PredictionResponse(BaseModel):
    predicted_price: float
    input_features: dict

# Frontend Page foe end User 
@app.get("/", response_class=HTMLResponse)
async def frontend():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>House Price Prediction</title>
        <link href="https://fonts.googleapis.com/css2?family=Roboto:wght@400;700&display=swap" rel="stylesheet">
        <style>
            body {
                font-family: 'Roboto', sans-serif;
                background: linear-gradient(135deg, #6a11cb 0%, #2575fc 100%);
                display: flex;
                justify-content: center;
                align-items: center;
                height: 100vh;
                margin: 0;
            }
            .card {
                background: white;
                padding: 30px;
                border-radius: 15px;
                box-shadow: 0 4px 20px rgba(0,0,0,0.2);
                max-width: 400px;
                width: 100%;
            }
            h1 { color: #2575fc; text-align: center; }
            input, button {
                width: 100%;
                padding: 10px;
                margin: 8px 0;
                border-radius: 8px;
                border: 1px solid #ccc;
            }
            button {
                background: #2575fc;
                color: white;
                border: none;
                cursor: pointer;
                font-size: 1.1em;
            }
            button:hover { background: #1a5edc; }
            #result { margin-top: 10px; text-align: center; font-weight: bold; }
        </style>
    </head>
    <body>
        <div class="card">
            <h1>🏠 Predict Price</h1>
            <form id="predictionForm">
                <input type="number" step="any" name="MedInc" placeholder="Median Income" required>
                <input type="number" step="any" name="HouseAge" placeholder="House Age" required>
                <input type="number" step="any" name="AveRooms" placeholder="Average Rooms" required>
                <input type="number" step="any" name="AveBedrms" placeholder="Average Bedrooms" required>
                <input type="number" step="any" name="AvePop" placeholder="Average Population" required>
                <button type="submit">Predict Price</button>
            </form>
            <div id="result"></div>
        </div>
        <script>
            document.getElementById("predictionForm").addEventListener("submit", async function(e) {
                e.preventDefault();
                const formData = new FormData(e.target);
                const data = Object.fromEntries(formData.entries());
                for (let key in data) data[key] = parseFloat(data[key]);
                
                const response = await fetch("/predict", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify(data)
                });
                const result = await response.json();
                if (response.ok) {
                    document.getElementById("result").textContent = "Predicted Price: $" + result.predicted_price.toFixed(2);
                    document.getElementById("result").style.color = "green";
                } else {
                    document.getElementById("result").textContent = "Error: " + result.detail;
                    document.getElementById("result").style.color = "red";
                }
            });
        </script>
    </body>
    </html>
    """


# Prediction API
@app.post("/predict", response_model=PredictionResponse)
async def predict_house_price(features: HouseFeatures):
    try:
        input_data = np.array([[
            features.MedInc, features.HouseAge, features.AveRooms,
            features.AveBedrms, features.AvePop
        ]])

        if input_data.shape[1] != scaler.n_features_in_:
            raise ValueError(f"Scaler expects {scaler.n_features_in_} features, got {input_data.shape[1]}")

        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)[0]
        predicted_price = prediction * 100000  # scale price

        return PredictionResponse(
            predicted_price=predicted_price,
            input_features=features.dict()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

# Run House Pric Prediction End to End Produc

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8500)

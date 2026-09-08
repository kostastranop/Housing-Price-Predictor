"""
California Housing Price Predictor — Flask Backend
Replicates the preprocessing pipeline from analysis.ipynb and serves predictions
from the saved RandomForestRegressor (productionmodel.sav).
"""

import os
import pickle
import logging

import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request
from flask_cors import CORS
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
app = Flask(__name__)
CORS(app)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants — feature order must match the training pipeline exactly
# ---------------------------------------------------------------------------
OCEAN_CATEGORIES = ["<1H OCEAN", "INLAND", "ISLAND", "NEAR BAY", "NEAR OCEAN"]

FEATURE_COLUMNS = [
    "longitude",
    "latitude",
    "housing_median_age",
    "total_rooms",
    "total_bedrooms",
    "population",
    "households",
    "median_income",
    "<1H OCEAN",
    "INLAND",
    "ISLAND",
    "NEAR BAY",
    "NEAR OCEAN",
    "bedroom_ratio",
    "household_rooms",
]

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "productionmodel.sav")
DATA_PATH = os.path.join(BASE_DIR, "housing.csv")

# ---------------------------------------------------------------------------
# Preprocessing helpers (mirroring analysis.ipynb)
# ---------------------------------------------------------------------------

def fix_distribution(data: pd.DataFrame) -> None:
    """Apply log(x+1) transform to skewed columns — in-place."""
    for col in ["total_rooms", "total_bedrooms", "population", "households"]:
        data[col] = np.log(data[col] + 1)


def one_hot_encode(data: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode ocean_proximity and ensure all 5 categories are present."""
    dummies = pd.get_dummies(data["ocean_proximity"])
    # Guarantee all training categories exist (handles unseen categories gracefully)
    for cat in OCEAN_CATEGORIES:
        if cat not in dummies.columns:
            dummies[cat] = False
    dummies = dummies[OCEAN_CATEGORIES]  # enforce column order
    return data.drop(columns=["ocean_proximity"]).join(dummies)


def feature_engineer(data: pd.DataFrame) -> None:
    """Add derived features — in-place."""
    data["bedroom_ratio"] = data["total_bedrooms"] / data["total_rooms"]
    data["household_rooms"] = data["total_rooms"] / data["households"]


# ---------------------------------------------------------------------------
# Build scaler by replaying the notebook pipeline on the full dataset
# ---------------------------------------------------------------------------

def build_scaler() -> StandardScaler:
    """
    Fit a StandardScaler on the entire dataset using the same pipeline
    as the notebook (Cell 30). The notebook does not save the scaler,
    so we must re-fit it at startup.
    """
    logger.info("Fitting StandardScaler on full dataset …")
    df = pd.read_csv(DATA_PATH)

    # Fill missing total_bedrooms (same as notebook Cell 6)
    imputer = SimpleImputer()
    df["total_bedrooms"] = imputer.fit_transform(df[["total_bedrooms"]])

    fix_distribution(df)
    df = one_hot_encode(df)
    feature_engineer(df)

    x = df.drop(columns=["median_house_value"])[FEATURE_COLUMNS]

    scaler = StandardScaler()
    scaler.fit(x)
    logger.info("Scaler ready.")
    return scaler


# ---------------------------------------------------------------------------
# Load model + scaler once at startup
# ---------------------------------------------------------------------------
logger.info("Loading production model …")
model = pickle.load(open(MODEL_PATH, "rb"))
scaler = build_scaler()
logger.info("Model loaded. Ready to serve predictions.")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/health")
def health():
    return jsonify({"status": "ok", "model": "RandomForestRegressor", "features": len(FEATURE_COLUMNS)})


@app.route("/predict", methods=["POST"])
def predict():
    """
    Expects JSON body:
    {
        "longitude": float,
        "latitude": float,
        "housing_median_age": float,
        "total_rooms": float,
        "total_bedrooms": float,
        "population": float,
        "households": float,
        "median_income": float,
        "ocean_proximity": str  (one of the 5 categories)
    }
    Returns:
    {
        "predicted_price": float
    }
    """
    try:
        data = request.get_json(force=True)

        # Validate required keys
        required = [
            "longitude", "latitude", "housing_median_age",
            "total_rooms", "total_bedrooms", "population",
            "households", "median_income", "ocean_proximity",
        ]
        missing = [k for k in required if k not in data]
        if missing:
            return jsonify({"error": f"Missing fields: {missing}"}), 400

        if data["ocean_proximity"] not in OCEAN_CATEGORIES:
            return jsonify({
                "error": f"ocean_proximity must be one of {OCEAN_CATEGORIES}"
            }), 400

        # Build a single-row DataFrame
        row = pd.DataFrame([{
            "longitude":          float(data["longitude"]),
            "latitude":           float(data["latitude"]),
            "housing_median_age": float(data["housing_median_age"]),
            "total_rooms":        float(data["total_rooms"]),
            "total_bedrooms":     float(data["total_bedrooms"]),
            "population":         float(data["population"]),
            "households":         float(data["households"]),
            "median_income":      float(data["median_income"]),
            "ocean_proximity":    data["ocean_proximity"],
        }])

        # Apply preprocessing pipeline
        fix_distribution(row)
        row = one_hot_encode(row)
        feature_engineer(row)

        # Align column order exactly as training
        x = row[FEATURE_COLUMNS].to_numpy()

        # Scale
        x_scaled = scaler.transform(x)

        # Predict
        prediction = model.predict(x_scaled)[0]
        predicted_price = max(0.0, float(prediction))  # clip negative edge cases

        logger.info(f"Prediction: ${predicted_price:,.0f}")
        return jsonify({"predicted_price": predicted_price})

    except (ValueError, TypeError) as e:
        return jsonify({"error": f"Invalid input: {str(e)}"}), 400
    except Exception as e:
        logger.exception("Unexpected error during prediction")
        return jsonify({"error": "Internal server error"}), 500


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True, port=5000)


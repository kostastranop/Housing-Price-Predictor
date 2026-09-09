# 🏠 California Housing Price Predictor

A full-stack machine learning application that predicts California housing prices using a **Random Forest Regressor** trained on the [California Housing dataset](https://www.kaggle.com/datasets/camnugent/california-housing-prices). The project includes an exploratory data analysis notebook, a production-ready model pipeline, and a Flask web application with a clean frontend interface.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [⚠️ Important: Model File Not Included](#️-important-model-file-not-included)
- [Setup & Installation](#setup--installation)
- [Step 1 — Generate the Model (Run the Notebook)](#step-1--generate-the-model-run-the-notebook)
- [Step 2 — Run the Web Application](#step-2--run-the-web-application)
- [API Reference](#api-reference)
- [Feature Engineering](#feature-engineering)

---

## Overview

Given a set of block-level housing attributes (location, age, room counts, population, income, and ocean proximity), the model predicts the **median house value** for a California district. The trained model is served via a Flask REST API and consumed by an interactive web frontend.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Data Analysis & Training | Python, Jupyter Notebook, scikit-learn, pandas, NumPy |
| Visualisation | Matplotlib, Seaborn |
| Backend / API | Flask, Flask-CORS |
| Frontend | HTML / CSS / JavaScript (Jinja2 template) |
| Model Serialisation | Pickle (`.sav`) |

---

## Project Structure

```
Housing Prices/
├── analysis.ipynb        # EDA, preprocessing pipeline, model training & export
├── app.py                # Flask backend — serves predictions from the saved model
├── housing.csv           # Raw California Housing dataset
├── requirements.txt      # Python dependencies
├── templates/
│   └── index.html        # Frontend web interface
└── productionmodel.sav   # ⚠️ NOT included — must be generated locally (see below)
```

---

## ⚠️ Important: Model File Not Included

The trained model file **`productionmodel.sav`** (~189 MB) is **not committed to this repository** due to its file size. It is listed in `.gitignore`.

> **You must run the Jupyter notebook (`analysis.ipynb`) yourself to train and export the model before the web app will function.**

Follow the steps below to do so.

---

## Setup & Installation

### Prerequisites

- Python **3.10+**
- `pip` (or a virtual environment manager such as `venv` or `conda`)

### Install Dependencies

Clone the repository and install all required packages:

```bash
git clone https://github.com/kostastranop/Housing-Price-Predictor.git
cd Housing-Price-Predictor
pip install -r requirements.txt
```

---

## Step 1 — Generate the Model (Run the Notebook)

This step trains the Random Forest model and saves it as `productionmodel.sav` in the project root. **This file is required before the app can start.**

1. **Launch Jupyter Notebook:**

   ```bash
   jupyter notebook
   ```

2. **Open `analysis.ipynb`** in the browser tab that appears.

3. **Run all cells in order:**
   - In the menu bar, click **Kernel → Restart & Run All**, then confirm.
   - Wait for all cells to finish executing. This may take a few minutes depending on your hardware.

4. **Verify the model was saved:**
   - Once complete, confirm that a file named `productionmodel.sav` now exists in the project root directory.

   ```bash
   # Linux / macOS
   ls -lh productionmodel.sav

   # Windows (PowerShell)
   Get-Item productionmodel.sav | Select-Object Name, Length
   ```

> **Note:** The notebook fits a `StandardScaler` and a `RandomForestRegressor` on the full dataset. The scaler is re-fitted at app startup from `housing.csv` using the same pipeline, so only the model file needs to be persisted.

---

## Step 2 — Run the Web Application

Once `productionmodel.sav` exists, start the Flask development server:

```bash
python app.py
```

The server will start on **`http://127.0.0.1:5000`** by default. Open that URL in your browser to access the prediction interface.

**Expected startup output:**

```
INFO:__main__:Loading production model …
INFO:__main__:Fitting StandardScaler on full dataset …
INFO:__main__:Scaler ready.
INFO:__main__:Model loaded. Ready to serve predictions.
 * Running on http://127.0.0.1:5000
```

---

## API Reference

### `GET /health`

Returns the model status.

**Response:**
```json
{
  "status": "ok",
  "model": "RandomForestRegressor",
  "features": 15
}
```

---

### `POST /predict`

Accepts housing attributes and returns a predicted median house value.

**Request body (JSON):**

| Field | Type | Description |
|---|---|---|
| `longitude` | `float` | Block longitude |
| `latitude` | `float` | Block latitude |
| `housing_median_age` | `float` | Median age of houses in the block |
| `total_rooms` | `float` | Total number of rooms |
| `total_bedrooms` | `float` | Total number of bedrooms |
| `population` | `float` | Block population |
| `households` | `float` | Number of households |
| `median_income` | `float` | Median household income (tens of thousands USD) |
| `ocean_proximity` | `string` | One of: `<1H OCEAN`, `INLAND`, `ISLAND`, `NEAR BAY`, `NEAR OCEAN` |

**Example request:**
```json
{
  "longitude": -122.23,
  "latitude": 37.88,
  "housing_median_age": 41,
  "total_rooms": 880,
  "total_bedrooms": 129,
  "population": 322,
  "households": 126,
  "median_income": 8.3252,
  "ocean_proximity": "NEAR BAY"
}
```

**Response:**
```json
{
  "predicted_price": 452600.00
}
```

---

## Feature Engineering

The preprocessing pipeline (applied identically during training and inference) includes:

| Step | Description |
|---|---|
| **Imputation** | Missing `total_bedrooms` values are filled using mean imputation |
| **Log transform** | `total_rooms`, `total_bedrooms`, `population`, `households` are log(x+1) transformed to reduce skew |
| **One-hot encoding** | `ocean_proximity` is expanded into 5 binary columns |
| **Derived features** | `bedroom_ratio = total_bedrooms / total_rooms` and `household_rooms = total_rooms / households` |
| **Standard scaling** | All 15 features are standardised (zero mean, unit variance) |

---

## License

This project is open source and available under the [MIT License](LICENSE).

# Bitcoin Price Prediction using LSTM 📈

A deep learning model that predicts Bitcoin closing prices using **LSTM (Long Short-Term Memory)** networks, achieving an **R² score of 0.954** on test data. Includes full EDA, price forecasting for the next 30 days, and interactive visualizations.

---

## Features

- **LSTM-based Time Series Forecasting** — Predicts Bitcoin closing prices using sequential price patterns
- **30-Day Future Prediction** — Forecasts the next 30 days of Bitcoin prices beyond the dataset
- **Comprehensive EDA** — Month-wise open/close/high/low analysis across 2020, 2021, and 2022
- **Interactive Visualizations** — Plotly-based charts for price trends and prediction comparisons
- **Full Evaluation Suite** — RMSE, MSE, MAE, R², and Explained Variance metrics

---

## Tech Stack

| Component | Technology |
|---|---|
| Deep Learning Model | LSTM (TensorFlow / Keras) |
| Data Processing | Pandas, NumPy, MinMaxScaler |
| Visualization | Plotly, Matplotlib |
| Evaluation Metrics | scikit-learn |
| Language | Python (Jupyter Notebook) |

---

## Dataset

- **Source:** Yahoo Finance (`BTC-USD.csv`)
- **Period:** September 2014 – February 2022
- **Total Records:** 2,713 daily entries
- **Features:** Date, Open, High, Low, Close, Adj Close, Volume
- **Prediction Window:** Last 1 year of data (365 days) used for training and testing

---

## Setup & Installation

### Prerequisites
- Python 3.8+
- Jupyter Notebook

### Install Dependencies

```bash
pip install pandas numpy tensorflow scikit-learn matplotlib plotly
```

### Run the Notebook

```bash
git clone https://github.com/bhavanachinnari/Bitcoin_Price_Prediction.git
cd Bitcoin_Price_Prediction
jupyter notebook
```

> **Note:** Place `BTC-USD.csv` in the same directory as the notebook before running.

---

## How It Works

1. **EDA** — Analyzes month-wise Bitcoin open, close, high, and low prices across 2020–2022 using Plotly bar charts and line charts
2. **Preprocessing** — Extracts the last 365 days of closing prices and normalizes using `MinMaxScaler`
3. **Dataset Creation** — Converts time series into supervised learning format using a sliding window of **15 time steps**
4. **Train/Test Split** — 60% training (219 samples) / 40% testing (146 samples)
5. **LSTM Model** — Single-layer LSTM with 10 units + Dense output layer, trained for 200 epochs
6. **Evaluation** — Predictions inverse-transformed and evaluated against original prices
7. **Future Forecasting** — Recursively predicts the next **30 days** using the last 15 known values

---

## Model Architecture

```
Sequential(
  LSTM(10, input_shape=(None, 1), activation='relu'),
  Dense(1)
)
Optimizer: Adam
Loss: Mean Squared Error
Epochs: 200 | Batch Size: 32
```

---

## Results

| Metric | Train | Test |
|---|---|---|
| **RMSE** | 2014.88 | **1935.32** |
| **MSE** | 4,059,762 | 3,745,467 |
| **MAE** | 1573.89 | **1500.51** |
| **R² Score** | 0.9533 | **0.9541** |
| **Explained Variance** | 0.9533 | **0.9541** |

---

## Visualizations

- Month-wise open vs close price comparison (2020, 2021, 2022)
- Month-wise high vs low price comparison
- Full closing price trend with train/test predictions overlaid
- Last 15 days vs next 30 days forecast plot
- Training vs validation loss curve

---

## Dependencies

```
pandas
numpy
tensorflow
scikit-learn
matplotlib
plotly
```

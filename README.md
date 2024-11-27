# FOREX_BUY_LSTM
# **Forex Trading Automation with LSTM**

This repository contains a Python script for automating Forex trading using **MetaTrader 5 (MT5)** and a **Long Short-Term Memory (LSTM)** model. The script predicts trading opportunities for a predefined list of Forex pairs and executes BUY orders based on the predictions.

---

## **Features**

- **Automated Data Retrieval**:
  - Fetches historical price data for multiple Forex pairs directly from MetaTrader 5.
  - Supports custom symbol lists.

- **Data Preprocessing**:
  - Generates technical indicators and sequences for time-series analysis.
  - Normalizes data for better model performance.

- **Machine Learning**:
  - Implements an LSTM model for predicting BUY opportunities.
  - Custom loss function using **Focal Loss** to handle class imbalance.

- **Trading Execution**:
  - Places BUY orders on MetaTrader 5 for symbols meeting precision thresholds.
  - Configurable risk management with stop-loss and take-profit levels.

- **Evaluation**:
  - Generates classification reports and confusion matrices for model predictions.

---

## **Requirements**

- **MetaTrader 5 (MT5)** installed and configured.
- **Python 3.8+** with the following libraries:
  - `MetaTrader5`
  - `pandas`
  - `numpy`
  - `torch`
  - `scikit-learn`
  - `imblearn`
  - `pytz`

Install the required dependencies using pip:

```bash
pip install MetaTrader5 pandas numpy torch scikit-learn imblearn pytz
```
## **Usage Instructions**

### **1. Configure MetaTrader 5**
- Ensure your **MT5 terminal** is running.
- Add your desired Forex symbols (e.g., `EURUSD`, `GBPUSD`) to the **Market Watch**.

---

### **2. Modify the Script**
- Edit the `symbols` list in the script to specify the Forex pairs you want to analyze:

  ```python
  symbols = ["EURUSD", "GBPUSD", "AUDUSD", "NZDUSD", "USDJPY"]
Configure parameters:
Modify the following parameters to fit your requirements:
lookback_days
total_days
batch_size

3. Execution
Run the script:
Execute the following command in your terminal:

bash
Copy code
python forex_lstm_trading.py
Process:
The script performs the following steps:

Retrieves historical data for the specified Forex pairs.
Processes data into sequences and trains the LSTM model.
Evaluates the model on a test set and generates predictions.
Places BUY orders in MT5 if the prediction meets specified thresholds.

Model Details
LSTM Architecture
Structure:
Two LSTM layers with:
Dropout for regularization.
Batch normalization for stable training.
Fully connected layer for binary classification.
Activation function:
Sigmoid.
Loss Function
Focal Loss: Designed to handle imbalanced target classes effectively.
Input Features
The model uses the following features:
Open
High
Low
Close
Volume

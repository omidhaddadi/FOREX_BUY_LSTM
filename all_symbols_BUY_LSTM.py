import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import RandomOverSampler
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime, timedelta
import pytz
import time

# Connect to MetaTrader 5
if not mt5.initialize():
    print("Failed to initialize MetaTrader5, error code:", mt5.last_error())
    quit()

# Define the list of symbols
symbols = ["EURUSD", "GBPUSD", "AUDUSD", "NZDUSD", "USDJPY", "USDCAD", "USDCHF", "EURJPY", "GBPAUD", "AUDNZD", "CADJPY", "GBPNZD", "AUDJPY", "EURGBP"]
#symbols = ["GBPUSD"]
#symbols = ["USDZAR","USDTRY","USDTHB","USDSGD","USDSEK"]
# Define the list to store predictions for each symbol
predictions = []

# Define the LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMModel, self).__init__()
        self.lstm1 = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.dropout1 = nn.Dropout(0.1)  # Reduced dropout rate
        self.bn1 = nn.BatchNorm1d(hidden_size)  # Batch normalization
        self.lstm2 = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.dropout2 = nn.Dropout(0.1)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        lstm_out, _ = self.lstm1(x)
        lstm_out = self.dropout1(lstm_out)
        lstm_out = self.bn1(lstm_out[:, -1, :])  # Apply batch normalization
        lstm_out, _ = self.lstm2(lstm_out.unsqueeze(1))
        lstm_out = self.dropout2(lstm_out)
        lstm_out = self.bn2(lstm_out[:, -1, :])
        out = self.fc(lstm_out)  # Only take the output of the last time step
        return self.sigmoid(out)

# Focal Loss implementation
class FocalLoss(nn.Module):
    def __init__(self, alpha=15, gamma=1.5, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = nn.BCEWithLogitsLoss(reduction='none')(inputs, targets)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        else:
            return F_loss

# Parameters
input_size = 5  # Number of features (Open, High, Low, Close, Volume)
hidden_size = 200
output_size = 1
learning_rate = 0.002
num_epochs = 250
batch_size = 500
# Define parameters
lookback_days = 20
total_days = 500
train_end = 400  # Training ends at day 400
test_start = 401  # Testing starts at day 401
test_end = 500  # Testing ends at day 500

for symbol in symbols:
    print(f"Processing symbol: {symbol}")

    # Request historical data from MetaTrader 5
    rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_D1, 0, total_days)
    if rates is None:
        print(f"Failed to retrieve data for {symbol}")
        continue

    # Convert to DataFrame
    data = pd.DataFrame(rates)
    data['time'] = pd.to_datetime(data['time'], unit='s')
    data.rename(columns={'time': 'Date', 'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'tick_volume': 'Volume'}, inplace=True)

    # Calculate Buy_Target
    data['Buy_Target'] = 0

    # Condition 1: The next 3 days' high price is at least 0.5% higher than the current day's close price
    high_condition = (data['High'].shift(-1).rolling(window=3).max() / data['Close'] > 1.005)

    # Condition 2: The next 3 days' low price is at most 0.5% lower than the current day's close price
    low_condition = (1 - (data['Low'].shift(-1).rolling(window=3).min() / data['Close']) < 0.005)

    # Combine the conditions
    data.loc[high_condition & low_condition, 'Buy_Target'] = 1

    # Handle missing values for the last few rows where future data is not available
    data['Buy_Target'].fillna(-1, inplace=True)

    # Feature selection
    X = data[['Open', 'High', 'Low', 'Close', 'Volume']].to_numpy()
    y = data['Buy_Target'].to_numpy()

    # Create sequences for training and testing
    def create_sequences(X, y, start, end, lookback):
        X_seq, y_seq = [], []
        for i in range(start, end + 1):
            if i - lookback < 0:  # Ensure we do not access negative indices
                continue
            X_seq.append(X[i - lookback:i])
            y_seq.append(y[i])
        return np.array(X_seq), np.array(y_seq)

    # Training data: day 21 to day 400
    X_train, y_train = create_sequences(X, y, lookback_days, train_end - 1, lookback_days)


    # Testing data: day 421 to day 500
    X_test, y_test = create_sequences(X, y, test_start, test_end - 1, lookback_days)


    # Normalize the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train.reshape(-1, X_train.shape[2])).reshape(X_train.shape)
    X_test = scaler.transform(X_test.reshape(-1, X_test.shape[2])).reshape(X_test.shape)

    # Convert data to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

    # Model training
    model = LSTMModel(input_size=X_train.shape[2], hidden_size=hidden_size, output_size=output_size)
    criterion = FocalLoss(alpha=15, gamma=1.5)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 5 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")

    # Evaluate the model
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test_tensor)
        y_pred_class = (y_pred > 0.65).float()

    # Classification report and confusion matrix
    y_test_np = y_test_tensor.numpy()
    y_pred_np = y_pred_class.numpy()

    
    report = classification_report(y_test_np, y_pred_np, output_dict=True, zero_division=0)
    confusion = confusion_matrix(y_test_np, y_pred_np)
    
    print("Classification Report:")
    print(classification_report(y_test_np, y_pred_np, zero_division=0))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test_np, y_pred_np))

    # Predict Buy_Target for the next day in the test set
    print(f"Final Test Predictions for {symbol}:")
    for i in range(len(y_pred_class)):
        print(f"Day {test_start + i}: Prediction = {y_pred_class[i].item()}")
    
    future_prediction_class = y_pred_class[i].item()
    
    # Handle cases where class '1' may not be present in the report
    precision_class_1 = report['1.0']['precision'] if '1.0' in report else 0

    # Store the result for this symbol
    predictions.append({
        'Symbol': symbol,
        'Precision_Class_1': precision_class_1,
        'Future_Prediction': int(future_prediction_class)
    })

    # Place a Buy order if the future prediction is 1
    if future_prediction_class == 1 and precision_class_1 >= 0.7:
        lot_size = 0.01  # Define the lot size
        price_info = mt5.symbol_info_tick(symbol)
        price = price_info.ask
        sl = price - (price * 0.005)  # Stop loss at 0.5% below
        tp = price + (price * 0.005)  # Take profit at 0.5% above

        # Create the request to open a Buy position
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": lot_size,
            "type": mt5.ORDER_TYPE_BUY,
            "price": price,
            "sl": sl,
            "tp": tp,
            "deviation": 20,  # Max price deviation in points
            "magic": 234000,  # Unique identifier for this order
            "comment": "Buy order",
            "type_time": mt5.ORDER_TIME_GTC,  # Good till canceled
            "type_filling": mt5.ORDER_FILLING_FOK,  # Fill or Kill filling mode
        }

        # Send the trade request
        result = mt5.order_send(request)

        # Check if the order was successfully placed
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            print(f"Failed to place buy order for {symbol}, error code: {result.retcode}")
        else:
            print(f"Buy order placed successfully for {symbol}, TP: {tp}, SL: {sl}")
    
# Shutdown MetaTrader 5
mt5.shutdown()

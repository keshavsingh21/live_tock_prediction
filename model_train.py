import yfinance as yf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
import joblib

# ✅ Use multiple stocks
stocks = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA","META","NFLX","JPM","GS"]

all_data = []

for stock in stocks:
    df = yf.download(stock, period="5y")[['Open','High','Low','Close','Volume']]
    all_data.append(df.values)

# Combine all stocks
data = np.vstack(all_data)

# Scale
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

# Create dataset
def create_dataset(data, timestep=60):
    X, y = [], []
    for i in range(60, len(data)):
        X.append(data[i-60:i])
        y.append(data[i][[0,3]])
    return np.array(X), np.array(y)

X, y = create_dataset(scaled_data)
X = X.reshape(X.shape[0], -1)

# Model
model = Sequential([
    Input(shape=(X.shape[1],)),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(2)
])


model.compile(optimizer='adam', loss=MeanSquaredError())
model.fit(X, y, epochs=20, batch_size=32)

# Save
model.save("model.keras")
joblib.dump(scaler, "scaler.pkl")


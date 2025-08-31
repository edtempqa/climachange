import os
import joblib
import torch
import torch.nn as nn
import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.base import BaseEstimator
from typing import Tuple
from cds import *
from configuration import *

logging.basicConfig(stream=sys.stdout, level=logging.INFO)

GRIB_FORMAT = 'grib'
PRODUCT_TYPE_MONTHLY_MEAN = 'monthly_mean'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Running on: '%s'", device)


def load_variable(short_name: str) -> pd.DataFrame:
    """Load and convert a single variable from GRIB to DataFrame."""
    try:
        ds = xr.open_dataset(DATA_FILE, engine=ENGINE, backend_kwargs={"filter_by_keys": {"shortName": short_name}})
        var_name = list(ds.data_vars)[0]
        df = ds.to_dataframe().reset_index()[[TIME_FIELD, var_name]]
        return df.rename(columns={var_name: short_name})
    except Exception as e:
        logging.error(f"Error loading variable '%s': %s", short_name, e)
        raise


def process_data() -> pd.DataFrame:
    """Preprocess and merge temperature and CO2 data into a monthly time series."""
    try:
        df_temp = load_variable(TEMPERATURE_FIELD_KELVIN)
        df_co2 = load_variable(CO2_FIELD)

        df_temp[TIME_FIELD] = pd.to_datetime(df_temp[TIME_FIELD])
        df_co2[TIME_FIELD] = pd.to_datetime(df_co2[TIME_FIELD])

        df_temp = df_temp.set_index(TIME_FIELD).resample("MS").mean().reset_index()
        df_co2 = df_co2.set_index(TIME_FIELD).resample("MS").mean().reset_index()

        df_merged = pd.merge(df_temp, df_co2, on=TIME_FIELD).dropna()

        df_merged[TEMPERATURE_FIELD_CELSIUS] = df_merged[TEMPERATURE_FIELD_KELVIN] - 273.15
        df_merged[CO2_FIELD_PPM] = df_merged[CO2_FIELD] * 1_000_000

        df_merged.set_index(TIME_FIELD, inplace=True)
        return df_merged[[TEMPERATURE_FIELD_CELSIUS, CO2_FIELD_PPM]]
    except Exception as ex:
        logging.error(f"Error processing data: '%s'", ex)
        raise


def save_model_and_scaler(model: nn.Module, scaler: BaseEstimator) -> None:
    """Save the LSTM model and the scaler."""
    try:
        torch.save(model.state_dict(), MODEL_PATH)
        joblib.dump(scaler, SCALER_PATH)
        print(f"Model saved to '{MODEL_PATH}' and scaler to '{SCALER_PATH}'.")
    except Exception as e:
        logging.error(f"Error saving model or scaler: '%s'", e)
        raise


def load_model_and_scaler() -> Tuple[nn.Module, BaseEstimator]:
    """Load the LSTM model and the scaler."""
    try:
        model = LSTMModel().to(device)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        scaler = joblib.load(SCALER_PATH)
        logging.info("Model and scaler loaded.")
        return model, scaler
    except Exception as e:
        logging.error(f"Error loading model or scaler: '%s'", e)
        raise


# ============== SEQUENCE CREATION ==============
def create_sequences(data: np.ndarray, seq_length: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Transform data into input-output sequences for LSTM."""
    x, y = [], []
    for i in range(len(data) - seq_length - N_PREDICT):
        x.append(data[i:i + seq_length])
        y.append(data[i + seq_length:i + seq_length + N_PREDICT])
    return (
        torch.tensor(np.array(x), dtype=torch.float32).to(device),
        torch.tensor(np.array(y), dtype=torch.float32).to(device)
    )


# Update LSTMModel to predict N_PREDICT steps ahead
class LSTMModel(nn.Module):
    def __init__(self, input_size: int = 2, hidden_size: int = 128,
                 num_layers: int = 1, output_size: int = 2, n_predict: int = N_PREDICT):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size * n_predict)
        self.n_predict = n_predict
        self.output_size = output_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(x)
        out = self.fc(lstm_out[:, -1, :])
        # Reshape to (batch, n_predict, output_size)
        return out.view(-1, self.n_predict, self.output_size)


def train_model(model: nn.Module, x_tensor: torch.Tensor, y_tensor: torch.Tensor) -> None:
    """Train the LSTM model to predict all N_PREDICT future steps."""
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.009)

    for epoch in range(EPOCHS):
        model.train()
        output = model(x_tensor)  # shape: (batch, N_PREDICT, 2)
        target = y_tensor  # shape: (batch, N_PREDICT, 2)
        loss = criterion(output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 20 == 0:
            logging.info(f"Epoch '%s'/'%s' - Loss: '%.6f'", epoch, EPOCHS, loss.item())


def predict(model: nn.Module, scaler: BaseEstimator, data_scaled: np.ndarray, dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Predict future values for N_PREDICT steps ahead."""
    model.eval()
    start_idx = -(SEQ_LENGTH + N_PREDICT)
    input_seq = data_scaled[start_idx:start_idx + SEQ_LENGTH]
    predictions = []

    for _ in range(N_PREDICT):
        input_tensor = torch.tensor(input_seq, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            pred = model(input_tensor).cpu().numpy()[0, 0, :]  # Take only the first predicted step
        predictions.append(pred)
        input_seq = np.vstack((input_seq[1:], pred.reshape(1, -1)))

    predictions = scaler.inverse_transform(np.array(predictions))
    return pd.DataFrame(predictions, columns=[TEMPERATURE_FIELD_CELSIUS, CO2_FIELD_PPM], index=dates)


# ============== VISUALIZATION ==============
def plot_predictions(dates: pd.DatetimeIndex, real: pd.DataFrame, predicted: pd.DataFrame) -> None:
    """Display temperature and CO2 real vs predicted."""
    plt.figure(figsize=(12, 6))

    # Temperature
    plt.plot(dates, real[TEMPERATURE_FIELD_CELSIUS], label="Actual temperature", color="tab:blue", marker='o')
    plt.plot(dates, predicted[TEMPERATURE_FIELD_CELSIUS], label="Predicted temperature", color="tab:blue",
             linestyle="--", marker='x')

    # CO2 (offset for scale visibility)
    plt.plot(dates, real[CO2_FIELD_PPM] - 600, label="CO2 real - 600", color="tab:green", marker='o')
    plt.plot(dates, predicted[CO2_FIELD_PPM] - 600, label="CO2 predicted - 600", color="tab:green",
             linestyle="--", marker='x')

    plt.title("Temperature and CO2 - Real vs Predicted")
    plt.xlabel("Time")
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


# ======= Main =======
def main(retrain=False):
    df = process_data()
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(df.values)
    x_tensor, y_tensor = create_sequences(data_scaled, SEQ_LENGTH)

    if retrain or not os.path.exists(MODEL_PATH):
        model = LSTMModel().to(device)
        train_model(model, x_tensor, y_tensor)
        save_model_and_scaler(model, scaler)
    else:
        model, scaler = load_model_and_scaler()

    comparison_dates = df.index[-N_PREDICT:]
    predicted = predict(model, scaler, data_scaled, comparison_dates)
    real = df.loc[comparison_dates]

    plot_predictions(comparison_dates, real, predicted)


def retrieve_data(data_format: str = GRIB_FORMAT) -> None:
    """Downloads the raw data from CDS source and store it."""
    builder = CDSRequestBuilder(DATA_SOURCE)
    # Example request for ERA5 monthly mean data on pressure level 1000 hPa
    request = (
        builder
        .set_variable([TIME_VARIABLE, CO2_VARIABLE])
        .set_pressure_level("1000")
        .set_area(44.6, 25.9, 44.3, 26.3)
        .set_year(["2003", "2004", "2005", "2006", "2007", "2008", "2009", "2010", "2011",
                   "2012", "2013", "2014", "2015", "2016", "2017", "2018", "2019", "2020"])
        .set_month(["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"])
        .set_data_format(data_format)
        .set_product_type([PRODUCT_TYPE_MONTHLY_MEAN])
        .build()
    )

    client = CDSClient()
    client.download_data(
        request=request,
        output_path=DATA_FILE
    )


if __name__ == "__main__":
    if DOWNLOAD_DATA:  # True to download data
        retrieve_data()
    main(retrain=RETRAIN)  # False to load existing model and scaler

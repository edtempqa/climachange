import os
from sklearn.preprocessing import MinMaxScaler
from data_loader import *
from visualization import *
from network import *
from configuration import *

logging.basicConfig(stream=sys.stdout, level=logging.INFO)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Running on: '%s'", device)

LABEL_ACTUAL_TEMP = "Actual temperature"
LABEL_PREDICTED_TEMP = "Predicted temperature"
LABEL_CO2_REAL = "CO2 real - 600"
LABEL_CO2_PREDICTED = "CO2 predicted - 600"
TITLE = "Temperature and CO2 - Real vs Predicted"

def main(retrain: bool = False) -> Tuple[pd.DatetimeIndex, pd.DataFrame, pd.DataFrame]:
    df = process_data(DATA_FILE)
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(df.values)
    x_tensor, y_tensor = create_sequences(data_scaled, SEQ_LENGTH, N_PREDICT, device)

    if retrain or not os.path.exists(MODEL_PATH):
        model = LSTMModel(n_predict=N_PREDICT).to(device)
        train_model(model, x_tensor, y_tensor, EPOCHS)
        save_model_and_scaler(model, scaler, MODEL_PATH, SCALER_PATH)
    else:
        model, scaler = load_model_and_scaler(device, MODEL_PATH, SCALER_PATH)

    comparison_dates = df.index[-N_PREDICT:]
    predicted = predict(model, scaler, data_scaled, N_PREDICT, SEQ_LENGTH, device)
    real = df.loc[comparison_dates]

    return comparison_dates, real, pd.DataFrame(predicted, columns=[TEMPERATURE_FIELD_CELSIUS, CO2_FIELD_PPM], index=comparison_dates)

if __name__ == "__main__":
    if DOWNLOAD_DATA:  # True to download data
        retrieve_data(DATA_SOURCE, DATA_FILE, GRIB_FORMAT)

    comparison_dates, real, predicted = main(retrain=RETRAIN) # False to load existing model and scaler

    temperature_real = SeriesDescriptor(data=real[TEMPERATURE_FIELD_CELSIUS], label=LABEL_ACTUAL_TEMP, color="tab:blue", marker='o')
    temperature_predicted = SeriesDescriptor(data=predicted[TEMPERATURE_FIELD_CELSIUS], label=LABEL_PREDICTED_TEMP, color="tab:blue", line_style="--", marker='x')
    co2_real = SeriesDescriptor(data=real[CO2_FIELD_PPM] - 600, label=LABEL_CO2_REAL, color="tab:green", marker='o')
    co2_predicted = SeriesDescriptor(data=predicted[CO2_FIELD_PPM] - 600, label=LABEL_CO2_PREDICTED, color="tab:green", line_style="--", marker='x')

    plot_series(comparison_dates, [temperature_real, temperature_predicted, co2_real, co2_predicted], TITLE)


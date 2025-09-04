import matplotlib.pyplot as plt
import pandas as pd
from configuration import *


LABEL_ACTUAL_TEMP = "Actual temperature"
LABEL_PREDICTED_TEMP = "Predicted temperature"
LABEL_CO2_REAL = "CO2 real - 600"
LABEL_CO2_PREDICTED = "CO2 predicted - 600"
TITLE = "Temperature and CO2 - Real vs Predicted"
X_LABEL = "Time"

def plot_predictions(dates: pd.DatetimeIndex, real: pd.DataFrame, predicted: pd.DataFrame) -> None:
    """
    Plot actual vs. predicted temperature and CO₂ values over time.

    This function generates a time series line plot comparing the actual and
    predicted values of temperature (in Celsius) and CO₂ concentration (in ppm)
    for the provided dates. CO₂ values are offset by -600 to make their scale
    comparable to temperature for visualization purposes. Actual values are
    plotted with solid lines and circle markers, while predicted values use
    dashed lines with cross markers.

    Args:
        dates (pd.DatetimeIndex): Sequence of timestamps corresponding to the data points.
        real (pd.DataFrame): DataFrame containing the actual measured values.
            Must include columns:
                - TEMPERATURE_FIELD_CELSIUS (temperature in °C)
                - CO2_FIELD_PPM (CO₂ concentration in ppm)
        predicted (pd.DataFrame): DataFrame containing the predicted values with the
            same required columns as `real`.

    Returns:
        None: Displays the plot directly using matplotlib.
    """
    plt.figure(figsize=(12, 6))

    # Temperature
    plt.plot(dates, real[TEMPERATURE_FIELD_CELSIUS], label=LABEL_ACTUAL_TEMP, color="tab:blue", marker='o')
    plt.plot(dates, predicted[TEMPERATURE_FIELD_CELSIUS], label=LABEL_PREDICTED_TEMP, color="tab:blue",
             linestyle="--", marker='x')

    # CO2 (offset for scale visibility)
    plt.plot(dates, real[CO2_FIELD_PPM] - 600, label=LABEL_CO2_REAL, color="tab:green", marker='o')
    plt.plot(dates, predicted[CO2_FIELD_PPM] - 600, label=LABEL_CO2_PREDICTED, color="tab:green",
             linestyle="--", marker='x')

    plt.title(TITLE)
    plt.xlabel(X_LABEL)
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
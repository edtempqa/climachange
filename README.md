# Temperature and CO2 Prediction

This project trains an RNN neural network (based on LSTM module) and tries to predict future temperature and CO2 levels using, 
based on monthly reanalysis data from the [Copernicus Climate Data Store (CDS)](https://ads.atmosphere.copernicus.eu/datasets/cams-global-ghg-reanalysis-egg4?tab=overview)

## Setup

1. **Install dependencies:**
```shell
pip install -r requirements.txt
```
2. **Configure parameters:**
   Edit `config.yaml` to set data source, variables, and training options.

3. **Run the script:**
   - If `DOWNLOAD_DATA` is set to `True` in `config.yaml`, the script will download data from CDS.
   - The model will be trained or loaded based on the `RETRAIN` flag.

## Main Features

- Downloads and processes climate data from CDS.
- Preprocesses and merges temperature and CO2 data.
- Trains an LSTM model to predict future values.
- Visualizes actual vs predicted temperature and CO2.

## Requirements

See `requirements.txt` for all dependencies.

## Notes

- Ensure you have valid CDS API credentials (`.cdsapirc` file) in your home directory:
```properties
url: https://ads.atmosphere.copernicus.eu/api
key: <your-api-key>
```
- The project uses GRIB data format and requires the `cfgrib` engine.

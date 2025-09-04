import xarray as xr
import pandas as pd
import torch
import numpy as np
from typing import Tuple
from configuration import *
from cds import *

logging.basicConfig(stream=sys.stdout, level=logging.INFO)

GRIB_FORMAT = 'grib'
PRODUCT_TYPE_MONTHLY_MEAN = 'monthly_mean'

def load_variable(data_file: str, short_name: str) -> pd.DataFrame:
    """
    Load and convert a single meteorological variable from a GRIB file into a DataFrame.

    This function uses xarray to open a GRIB file and extract the specified
    variable (identified by its short name). The data is then converted into
    a Pandas DataFrame containing timestamps and the selected variable values.
    The variable column is renamed to match the provided short name for clarity.

    Args:
        data_file (str): Path to the GRIB file containing meteorological data.
        short_name (str): GRIB short name of the variable to extract (e.g., "2t" for temperature, "co2" for CO₂).

    Returns:
        pd.DataFrame: A DataFrame with two columns:
            - TIME_FIELD (datetime): The timestamp of each observation.
            - short_name (float): The extracted variable values.

    Raises:
        FileNotFoundError: If the specified GRIB file does not exist.
        Exception: If the file cannot be read or the variable cannot be extracted.
    """
    try:
        with xr.open_dataset(data_file, engine=ENGINE, backend_kwargs={"filter_by_keys": {"shortName": short_name}}) as ds:
            var_name = list(ds.data_vars)[0]
            df = ds.to_dataframe().reset_index()[[TIME_FIELD, var_name]]
            return df.rename(columns={var_name: short_name})
    except FileNotFoundError:
        logging.exception(f"Data file `{data_file}` not found.")
        raise
    except Exception:
        logging.exception(f"Error loading variable `{short_name}`.")
        raise

def process_data(data_file: str) -> pd.DataFrame:
    """
    Load, clean, and process climate data from a NetCDF or compatible file.

    This function extracts temperature and CO₂ data from the provided file,
    aligns them to a monthly frequency, merges them into a single dataset,
    and performs necessary unit conversions:
        - Temperature is converted from Kelvin to Celsius.
        - CO₂ mole fraction is converted to parts per million (ppm).

    The resulting DataFrame is indexed by time and contains only the processed
    temperature and CO₂ columns, ready for analysis or modeling.

    Args:
        data_file (str): Path to the input dataset file containing climate variables.

    Returns:
        pd.DataFrame: A DataFrame indexed by datetime with the following columns:
            - TEMPERATURE_FIELD_CELSIUS: Monthly mean temperature (°C).
            - CO2_FIELD_PPM: Monthly mean atmospheric CO₂ concentration (ppm).

    Raises:
        Exception: If data loading, processing, or merging fails. The error is
        logged before being re-raised.
    """
    try:
        df_temp = load_variable(data_file, TEMPERATURE_FIELD_KELVIN)
        df_co2 = load_variable(data_file, CO2_FIELD)

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

# ============== SEQUENCE CREATION ==============
def create_sequences(data: np.ndarray,
                     seq_length: int,
                     n_predict: int,
                     device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Transform a time series into input-output sequences suitable for LSTM training.

    This function slides a window of length `seq_length` over the input data to create
    input sequences, and pairs each with the following `n_predict` values as targets.
    The resulting arrays are converted to PyTorch tensors and moved to the specified device.

    Args:
        data (np.ndarray): 1D or 2D array containing the time series data.
        seq_length (int): Number of time steps in each input sequence.
        n_predict (int): Number of future steps to predict (output sequence length).
        device (torch.device): The device (CPU or GPU) where the resulting tensors will be stored.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            - x (torch.Tensor): Tensor of shape (num_samples, seq_length, ...)
                containing input sequences.
            - y (torch.Tensor): Tensor of shape (num_samples, n_predict, ...)
                containing target sequences.

    Example:
        >>> data = np.arange(10)  # [0,1,2,3,4,5,6,7,8,9]
        >>> x, y = create_sequences(data, seq_length=3, n_predict=2, device=torch.device("cpu"))
        >>> x.shape, y.shape
        (torch.Size([5, 3]), torch.Size([5, 2]))
    """
    x, y = [], []
    for i in range(len(data) - seq_length - n_predict):
        x.append(data[i:i + seq_length])
        y.append(data[i + seq_length:i + seq_length + n_predict])
    return (
        torch.tensor(np.array(x), dtype=torch.float32).to(device),
        torch.tensor(np.array(y), dtype=torch.float32).to(device)
    )

def retrieve_data(data_source: str, output_path: str, data_format: str) -> None:
    """
    Retrieve climate data from the Copernicus Climate Data Store (CDS) and save it locally.

    This function builds and executes a data retrieval request for ERA5 monthly mean
    reanalysis data at the 1000 hPa pressure level. It requests time and CO₂ variables
    over a predefined geographic bounding box and a fixed time span.
    The retrieved dataset is downloaded and saved in the specified format.

    Args:
        data_source (str): Name of the CDS dataset (e.g., "reanalysis-era5-pressure-levels-monthly-means").
        output_path (str): Path where the downloaded dataset will be stored.
        data_format (str): Desired output format (e.g., "netcdf", "grib").

    Returns:
        None: The dataset is downloaded and saved locally; no object is returned.

    Notes:
        - The request is hardcoded to use:
            * Variables: time and CO₂
            * Pressure level: 1000 hPa
            * Area: Latitude 44.6°N to 44.3°N, Longitude 25.9°E to 26.3°E
            * Years: 2003–2020
            * Months: January–December
            * Product type: monthly mean
        - Requires a configured CDS API key for authentication.
    """
    builder = CDSRequestBuilder(data_source)
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
        output_path=output_path
    )

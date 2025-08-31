import cdsapi
import zipfile
import logging
import sys
from typing import List
from typing import Union

logging.basicConfig(stream=sys.stdout, level=logging.INFO)

class CDSRequestBuilder:
    """
    Builder class for constructing request parameters for the CDS API.
    """

    def __init__(self, dataset_name: str):
        self._params = {}
        self.dataset = dataset_name

    def set_variable(self, variables: Union[str, List[str]]):
        self._params["variable"] = variables
        return self

    def set_year(self, year: Union[str, List[str]]):
        self._params["year"] = year
        return self

    def set_month(self, month: Union[str, List[str]]):
        self._params["month"] = month
        return self

    def set_pressure_level(self, level: str):
        self._params["pressure_level"] = level
        return self

    def set_model_level(self, model_level: str):
        self._params["model_level"] = model_level
        return self

    def set_step(self, step: str):
        self._params["step"] = step
        return self

    def set_product_type(self, product_type: Union[str, List[str]]):
        self._params["product_type"] = product_type
        return self

    def set_data_format(self, data_format: str):
        self._params["data_format"] = data_format
        return self

    def set_area(self, north: float, west: float, south: float, east: float):
        self._params["area"] = [north, west, south, east]
        return self

    def set_date_range(self, start: str, end: str):
        self._params["date"] = [f"{start}/{end}"]
        return self

    def set_exact_dates(self, years: Union[str, List[str]], months: Union[str, List[str]] = None, days: Union[str, List[str]] = None):
        self._params["year"] = years
        if months:
            self._params["month"] = months
        if days:
            self._params["day"] = days
        return self

    def set_time(self, time: str = "00:00"):
        self._params["time"] = time
        return self

    def build(self):
        return {
            "dataset": self.dataset,
            "params": self._params
        }


class CDSClient:
    """
    Client for interacting with the CDS API, downloading and extracting data.
    """
    def __init__(self):
        self.client = cdsapi.Client()

    def download_and_extract(self, request: dict, zip_file_path: str, extract_to: str = None) -> None:
        """
        Download data and extract ZIP file if specified.

        Args:
            request (dict): Request dictionary.
            zip_file_path (str): Path to save ZIP file.
            extract_to (str, optional): Directory to extract files.
        """

        try:
            self.download_data(request, output_path=zip_file_path)
        except Exception as e:
            logging.error(f"Failed to download data: {e}")
            return

        if extract_to and zipfile.is_zipfile(zip_file_path):
            try:
                with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
                    zip_ref.extractall(extract_to)
                    logging.info(f"Extracted to: '%s'", extract_to)
            except Exception as e:
                logging.error(f"Failed to extract ZIP file: {e}")

    def download_data(self, request: dict, output_path: str) -> None:
        """
        Download data from the CDS API.

        Args:
            request (dict): Request dictionary.
            output_path (str): Path to save downloaded file.
        """

        dataset = request["dataset"]
        params = request["params"]

        logging.info(f"Sending request to CDS API for dataset: '%s'", dataset)
        self.client.retrieve(dataset, params, output_path)
        logging.info(f"Download completed: '%s'", output_path)

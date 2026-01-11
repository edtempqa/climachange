import matplotlib.pyplot as plt
import pandas as pd
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class SeriesDescriptor:
    data: pd.Series = None
    label: Optional[str] = None
    color: Optional[str] = None
    marker: Optional[str] = None
    line_style: Optional[str] = None

def plot_series(dates: pd.DatetimeIndex, series_list: List[SeriesDescriptor], title: str) -> None:
    """
    Plot a list of series (for example temperature, CO₂, etc.) values over time.

    Args:
        dates (pd.DatetimeIndex): Sequence of timestamps corresponding to the data points.
        series_list (List[SeriesDescriptor]): a list of SeriesDescriptor objects containing the data to be displayed and styling info.
        title (str): Title of the plot.

    Returns:
        None: Displays the plot directly using matplotlib.
    """
    plt.figure(figsize=(12, 6))
    for desc in series_list:
        plt.plot(dates, desc.data, label=desc.label, color=desc.color, marker=desc.marker, linestyle=desc.line_style)
    plt.title(title)
    plt.xlabel("Time")
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
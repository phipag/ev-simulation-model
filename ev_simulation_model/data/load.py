from pathlib import Path
from typing import Union

import pandas as pd


def load_norway_residential_data(path: Union[str, Path]) -> pd.DataFrame:
    return pd.read_csv(
        path,
        index_col=False,
        sep=";",
        parse_dates=["Start_plugin", "End_plugout"],
        decimal=",",
    )

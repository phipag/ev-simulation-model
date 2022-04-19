import pandas as pd

from ev_simulation_model.data import load_norway_residential_data
from tests.conftest import TEST_DATA_PATH

MANDATORY_COLUMNS = ["El_kWh", "Duration_hours", "Start_plugin_hour"]


def assert_df_has_mandatory_columns(df: pd.DataFrame) -> None:
    assert all(col in df.columns for col in MANDATORY_COLUMNS)


def test_load_norway_residential_data_pathlib_ok():
    df = load_norway_residential_data(TEST_DATA_PATH / "Dataset 1_EV charging reports.csv")
    assert_df_has_mandatory_columns(df)


def test_load_norway_residential_data_path_string_ok():
    df = load_norway_residential_data(str(TEST_DATA_PATH / "Dataset 1_EV charging reports.csv"))
    assert_df_has_mandatory_columns(df)

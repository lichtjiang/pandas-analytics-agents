import os
from importlib.resources import as_file, files
from pathlib import Path

import pandas as pd


def normalize_covid_dataset(df: pd.DataFrame) -> pd.DataFrame:
    # normalize the column names for easier access when only a subset of columns is selected
    covid_df = df.rename(
        columns={
            "Country/Region": "country",
            "Deaths": "deaths",
            "Recovered": "recovered",
        }
    )
    return covid_df

# Note: if you need more than three columns, do NOT call normalize_covid_dataset() before the two return statements below,
# because it will rename the columns and you will lose access to other columns.
def load_covid_dataset() -> pd.DataFrame:
    configured_path = os.getenv("DATASET_COVID19")
    if configured_path:
        configured_path = Path(os.getenv("DATASET_COVID19", "./src/pandas_analytics_agents/dataset/covid19_country_wise_latest.csv"))
        print(f"Based on DATASET_COVID19, csv file is here: {os.path.abspath(configured_path)}")
        covid_df = pd.read_csv(configured_path)
        return normalize_covid_dataset(covid_df)

    resource = files("pandas_analytics_agents").joinpath( "dataset/covid19_country_wise_latest.csv")

    with as_file(resource) as csv_path:
        covid_df = pd.read_csv(csv_path)
        covid_df = normalize_covid_dataset(covid_df)
        return covid_df


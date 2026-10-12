import duckdb
import pandas as pd

from pandas_analytics_agents.utils.dataset import load_covid_dataset

COVID_TABLE_NAME = "covid_19"

def get_df_from_sql_clean(sql_clean: str, covid_df: pd.DataFrame | None) -> pd.DataFrame:
    
    # Load the COVID-19 dataset
    if covid_df is None:
        covid_df = load_covid_dataset()

    # Register the DataFrame with DuckDB, and execute the SQL query cleanly
    with duckdb.connect(database=":memory:") as con:
        con.register(COVID_TABLE_NAME, covid_df)
        df = con.execute(sql_clean).fetch_df()
    
    return df


def connect_and_register_df( covid_df: pd.DataFrame | None) -> duckdb.DuckDBPyConnection:
    if covid_df is None:
        covid_df = load_covid_dataset()

    # Register the DataFrame with DuckDB.
    duckdb_conn = duckdb.connect(database=":memory:")

    duckdb_conn.register(
        COVID_TABLE_NAME,
        covid_df,
    )

    return duckdb_conn

def disconnect_and_unregister_df(duckdb_conn: duckdb.DuckDBPyConnection):
    # Unregister the DataFrame and close the DuckDB connection.
    duckdb_conn.unregister(COVID_TABLE_NAME)
    duckdb_conn.close()


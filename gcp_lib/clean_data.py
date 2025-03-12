import pandas as pd
from google.cloud import bigquery
from colorama import Fore, Style
from pathlib import Path
from gcp_lib.params import *

def load_data_from_csv_to_dataframe(csv_path) -> pd.DataFrame:
    """
    Loads a CSV file to Pandas DataFrame.

    Args:
        csv_path (str): Path to the CSV file.
        dataframe (DataFrame): Object of Dataframe
    """
    print(Fore.BLUE + f"\n📂 Loading CSV from {csv_path}..." + Style.RESET_ALL)

    df = pd.read_csv(csv_path)

    print(f"✅ CSV loaded, shape: {df.shape}")

    return df

def upload_df_to_bigquery(df: pd.DataFrame, table: str, truncate: bool = True) -> None:
    """
    Uploads a CSV file to BigQuery with autodetected schema.

    Args:
        csv_path (str): Path to the CSV file.
        table (str): Name of the target BigQuery table.
        truncate (bool): Whether to overwrite (True) or append (False) data.
    """
    try:
        # Connect BQ
        client = bigquery.Client(project=GCP_PROJECT)

        # 🔥 Config BigQuery
        write_mode = "WRITE_TRUNCATE" if truncate else "WRITE_APPEND"
        job_config = bigquery.LoadJobConfig(
            write_disposition=write_mode,
            autodetect=True
        )

        full_table_name = f"{GCP_PROJECT}.{BQ_DATASET}.{table}"

        print(Fore.BLUE + f"\n🚀 Uploading data to BigQuery: {full_table_name}..." + Style.RESET_ALL)

        # 🔄 launch loading dataframe
        job = client.load_table_from_dataframe(df, full_table_name, job_config=job_config)
        job.result()

        print(Fore.GREEN + f"✅ Successfully uploaded to {full_table_name}, rows: {df.shape[0]}" + Style.RESET_ALL)

    except Exception as e:
        print(Fore.RED + f"❌ Error: {str(e)}" + Style.RESET_ALL)

def load_data_from_bigquery(table: str) -> pd.DataFrame:
    """
    Loads data from a specified BigQuery table into a Pandas DataFrame.

    Args:
        table (str): Name of the BigQuery table (in format 'dataset.table').

    Returns:
        pd.DataFrame: Dataframe containing the table data.
    """
    print(Fore.BLUE + f"\n🔄 Fetching data from BigQuery table: {table}..." + Style.RESET_ALL)

    try:
        # Initialize BigQuery client
        client = bigquery.Client(project=GCP_PROJECT)

        # Construct query
        query = f"SELECT * FROM `{GCP_PROJECT}.{BQ_DATASET}.{table}`"

        # Execute query
        query_job = client.query(query)
        df = query_job.to_dataframe()

        print(Fore.GREEN + f"✅ Data fetched successfully! Rows retrieved: {df.shape[0]}" + Style.RESET_ALL)
        return df

    except Exception as e:
        print(Fore.RED + f"❌ Error loading data from BigQuery: {str(e)}" + Style.RESET_ALL)
        return pd.DataFrame()

# for a possible future usage
def get_data_with_cache(gcp_project: str, query: str, cache_path: Path, data_has_header=True) -> pd.DataFrame:
    """
    Retrieve `query` data from BigQuery, or from `cache_path` if the file exists.
    """
    if cache_path.is_file():
        print(Fore.BLUE + "\nLoading data from local cache..." + Style.RESET_ALL)
        df = pd.read_csv(cache_path, header='infer' if data_has_header else None)
    else:
        print(Fore.BLUE + "\nFetching data from BigQuery..." + Style.RESET_ALL)
        client = bigquery.Client(project=gcp_project)
        query_job = client.query(query)
        df = query_job.result().to_dataframe()

        if df.shape[0] > 0:
            df.to_csv(cache_path, header=data_has_header, index=False)

    print(f"✅ Data loaded with shape {df.shape}")
    return df

def load_data_to_bq(data: pd.DataFrame, gcp_project: str, bq_dataset: str, table: str, truncate: bool) -> None:
    """
    Load DataFrame into BigQuery without predefined schema (auto-detect schema).
    """
    assert isinstance(data, pd.DataFrame)
    full_table_name = f"{gcp_project}.{bq_dataset}.{table}"
    print(Fore.BLUE + f"\nSaving data to BigQuery @ {full_table_name}..." + Style.RESET_ALL)

    client = bigquery.Client()

    # Nastavení režimu zápisu - přepis tabulky nebo přidání dat
    write_mode = "WRITE_TRUNCATE" if truncate else "WRITE_APPEND"
    job_config = bigquery.LoadJobConfig(
        write_disposition=write_mode,
        autodetect=True,  # 🔥 Nech BigQuery, aby si schéma určil sám
        source_format=bigquery.SourceFormat.CSV,  # 📌 Řekneme BigQuery, že se jedná o CSV
        skip_leading_rows=1  # 📌 Přeskakujeme první řádek jako hlavičku (pokud je tam)
    )

    print(f"\n{'Overwriting' if truncate else 'Appending'} {full_table_name} ({data.shape[0]} rows)")

    # Nahrání do BigQuery
    job = client.load_table_from_dataframe(data, full_table_name, job_config=job_config)
    job.result()  # Počkáme na dokončení jobu

    print(f"✅ Data uploaded to BigQuery with shape {data.shape}")

def clean_data_rotten_tomatoes():
    """
    Clean and upload cleaned data for both movie reviews and movie details.
    """
    datasets = [
        (TABLE_DATA_TOMATO_MOVIES_RAW, TABLE_DATA_TOMATO_MOVIES_CLEANED, "cached_cleaned_movies")
    ]

    for raw_table, cleaned_table, cache_filename in datasets:
        print(f"\nProcessing {raw_table}...")

        # Query raw data
        query = f"SELECT * FROM `{GCP_PROJECT}.{BQ_DATASET}.{raw_table}`"
        df = get_data_with_cache(GCP_PROJECT, query, Path(f"raw_data/kaggle_rotten_tomatoes/{cache_filename}.csv"))

        # Data cleaning steps
        df = df.drop_duplicates()
        df = df.dropna(how='any')

        print("✅ Data cleaned")

        # Upload cleaned data
        load_data_to_bq(df, GCP_PROJECT, BQ_DATASET, cleaned_table, True)
        print("✅ Cleaned data uploaded")

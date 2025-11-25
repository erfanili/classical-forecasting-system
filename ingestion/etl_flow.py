from ingest_raw import load_all_regions
from detect_anomalies import detect_spikes
from validate_schema import validate, clean
from write_parquet import save_cleaned_data


RAW_PATH = 'data/raw'
CLEAN_PATH = 'data/cleaned/clean_data.parquet'
df = load_all_regions(path=RAW_PATH)
df = validate(df)
df = clean(df)
df = detect_spikes(df)
df = save_cleaned_data(df,path=CLEAN_PATH)
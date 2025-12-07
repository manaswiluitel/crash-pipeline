# cleaning_rules.py
import pandas as pd
import json

def clean_for_gold(df: pd.DataFrame, lat_col="latitude", lng_col="longitude") -> pd.DataFrame:
    """
    Cleans a merged crash dataset for insertion into DuckDB gold table.

    Args:
        df: Merged raw dataframe (crashes + vehicles + people)
        lat_col: Name of latitude column
        lng_col: Name of longitude column

    Returns:
        cleaned DataFrame ready for DuckDB insertion
    """

    df = df.copy()

    # Ensure crash_record_id exists
    if "crash_record_id" not in df.columns:
        raise KeyError("Input dataframe must have 'crash_record_id'")

    # Fill missing numeric values with 0
    for col in df.select_dtypes(include=["int64", "float64"]).columns:
        df[col] = df[col].fillna(0)

    # Fill missing strings with empty string
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].fillna("")

    # Normalize latitude / longitude to floats
    if lat_col in df.columns:
        df[lat_col] = pd.to_numeric(df[lat_col], errors="coerce")
    if lng_col in df.columns:
        df[lng_col] = pd.to_numeric(df[lng_col], errors="coerce")

    # Create JSON lists for vehicle make/model if vehicle columns exist
    if "veh_make" in df.columns:
        df["veh_make_list_json"] = df.groupby("crash_record_id")["veh_make"]\
                                     .transform(lambda x: json.dumps(list(x.unique())))
    if "veh_model" in df.columns:
        df["veh_model_list_json"] = df.groupby("crash_record_id")["veh_model"]\
                                      .transform(lambda x: json.dumps(list(x.unique())))

    # Drop duplicates by crash_record_id, keep first
    df = df.drop_duplicates(subset=["crash_record_id"])

    return df

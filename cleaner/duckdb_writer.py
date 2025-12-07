
# duckdb_writer.py
import duckdb
import pandas as pd

TYPE_MAP = {
    "int64": "BIGINT",
    "Int64": "BIGINT",
    "float64": "DOUBLE",
    "bool": "BOOLEAN",
    "boolean": "BOOLEAN",
    "object": "VARCHAR",
    "datetime64[ns]": "TIMESTAMP",
}

def _map_type(dtype: str) -> str:
    return TYPE_MAP.get(dtype, "VARCHAR")

def _parse_schema_table(name: str):
    """
    Accepts:
      - "schema.table"  (preferred)
      - "table"         (falls back to schema='main')
    Returns (schema, table)
    """
    parts = name.split(".")
    if len(parts) == 2:
        return parts[0], parts[1]
    elif len(parts) == 1:
        return "main", parts[0]
    else:
        raise ValueError(f"Invalid table identifier: {name}")

def ensure_schema_and_table(con: duckdb.DuckDBPyConnection, table: str, df: pd.DataFrame, pk: str = "crash_record_id"):
    schema, tbl = _parse_schema_table(table)

    # Ensure schema and set it active (no catalog ambiguity)
    con.execute(f'CREATE SCHEMA IF NOT EXISTS "{schema}";')
    con.execute(f"SET schema '{schema}';")

    # CREATE TABLE IF NOT EXISTS with columns from df
    cols_ddl = []
    for col in df.columns:
        duck_type = _map_type(str(df[col].dtype))
        cols_ddl.append(f'"{col}" {duck_type}')
    cols_ddl_str = ",\n  ".join(cols_ddl)

    con.execute(f"""
        CREATE TABLE IF NOT EXISTS "{tbl}" (
          {cols_ddl_str},
          PRIMARY KEY ("{pk}")
        );
    """)

    # Read existing column NAMES correctly from PRAGMA table_info (index 1)
    info = con.execute(f'PRAGMA table_info("{tbl}")').fetchall()
    existing_cols_lower = {str(r[1]).lower() for r in info}  # r[1] is column name

    # Add new columns from df only if missing (case-insensitive)
    for col in df.columns:
        if col.lower() not in existing_cols_lower:
            duck_type = _map_type(str(df[col].dtype))
            con.execute(f'ALTER TABLE "{tbl}" ADD COLUMN "{col}" {duck_type};')

def upsert_dataframe(con: duckdb.DuckDBPyConnection, table: str, df: pd.DataFrame, key: str = "crash_record_id") -> dict:
    if key not in df.columns:
        raise KeyError(f"Required key column '{key}' not in dataframe")

    schema, tbl = _parse_schema_table(table)
    con.execute(f"SET schema '{schema}';")

    # Register DataFrame as a temp view
    con.register("cleaned", df)

    # Existing table columns (for a safe, ordered insert column list)
    info = con.execute(f'PRAGMA table_info("{tbl}")').fetchall()
    table_cols = [str(r[1]) for r in info]  # names
    # Only use columns that exist in BOTH the table and the incoming df
    cols = [c for c in table_cols if c in df.columns]

    # Estimate how many would be NEW (for logging/idempotency proof)
    not_exists = con.execute(f'''
        SELECT COUNT(*)
        FROM cleaned c
        LEFT JOIN "{tbl}" t USING("{key}")
        WHERE t."{key}" IS NULL
    ''').fetchone()[0]

    before = con.execute(f'SELECT COUNT(*) FROM "{tbl}"').fetchone()[0]

    col_list = ", ".join([f'"{c}"' for c in cols])
    sel_list = ", ".join([f'c."{c}"' for c in cols])

    # Version-agnostic UPSERT: INSERT OR REPLACE (uses PRIMARY KEY uniqueness)
    con.execute(f'''
        INSERT OR REPLACE INTO "{tbl}" ({col_list})
        SELECT {sel_list} FROM cleaned c;
    ''')

    after = con.execute(f'SELECT COUNT(*) FROM "{tbl}"').fetchone()[0]
    # "inserted" == net new rows; updates won't change count
    return {"before": before, "after": after, "inserted": after - before, "estimated_new": not_exists}

def verify_sql(table: str) -> dict:
    schema, tbl = _parse_schema_table(table)
    return {
        "count":  f'SELECT COUNT(*) AS total FROM "{schema}"."{tbl}";',
        "dupes": (
            f'SELECT "crash_record_id", COUNT(*) AS n FROM "{schema}"."{tbl}" '
            'GROUP BY 1 HAVING COUNT(*) > 1 ORDER BY n DESC LIMIT 10;'
        ),
        "recent": f'SELECT * FROM "{schema}"."{tbl}" ORDER BY crash_date DESC NULLS LAST LIMIT 5;',
    }

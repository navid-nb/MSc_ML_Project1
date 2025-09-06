import os
import datetime as dt
import pandas as pd
import duckdb
import wrds
import pyarrow as pa
import pyarrow.parquet as pq

BASE_DIR = "wrds_extracts"

# ----------------------------
# Helpers
# ----------------------------
def list_runs(base_dir=BASE_DIR):
    if not os.path.isdir(base_dir):
        return []
    runs = [d for d in os.listdir(base_dir)
            if os.path.isdir(os.path.join(base_dir, d)) and d.startswith("run_")]
    runs.sort()
    return runs

def latest_run(base_dir=BASE_DIR):
    runs = list_runs(base_dir)
    return runs[-1] if runs else None

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def query_to_parquet(conn, sql_path, out_path, params=None, chunksize=500_000):
    sql = open(sql_path).read()
    params = params or {}
    writer = None
    for chunk in pd.read_sql_query(sql, con=conn.connection, params=params, chunksize=chunksize):
        table = pa.Table.from_pandas(chunk, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(out_path, table.schema)
        writer.write_table(table)
    if writer is not None:
        writer.close()

def views_from_parquet(con, view_name, file_path):
    con.execute(f"CREATE VIEW {view_name} AS SELECT * FROM read_parquet('{file_path}');")

# ----------------------------
# Main logic as a function
# ----------------------------
def wrds_extract_and_join(
    wrds_user="wboughattas",
    start="2000-01-01",
    end="2025-01-01",
    chunksize=500_000,
    use_run="last",   # "last", "new", or specific folder name
    migrations_dir="migrations",
    join_sql="migrations/010_final_join.sql"
):
    runs_base = BASE_DIR
    ensure_dir(runs_base)

    # Decide run folder based on use_run
    reuse = False
    if use_run == "new":
        stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        outdir_name = f"run_{stamp}"
    elif use_run == "last":
        last = latest_run(runs_base)
        if last:
            outdir_name = last
            reuse = True
        else:
            stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
            outdir_name = f"run_{stamp}"
    else:
        outdir_name = use_run
        reuse = os.path.isdir(os.path.join(runs_base, outdir_name))

    outdir = os.path.join(runs_base, outdir_name)
    ensure_dir(outdir)
    print(f"[info] Using run folder: {outdir_name} (reuse={reuse})")

    artifacts = [
        (os.path.join(migrations_dir, "001_base_extract.sql"),  "dsf.parquet"),
        (os.path.join(migrations_dir, "002_crsp_names.sql"),    "stocknames.parquet"),
        (os.path.join(migrations_dir, "003_comp_secm.sql"),     "secm.parquet"),
        (os.path.join(migrations_dir, "004_comp_fundq.sql"),    "fundq.parquet"),
        (os.path.join(migrations_dir, "005_ff_factors.sql"),    "ff.parquet"),
        (os.path.join(migrations_dir, "006_ibes_statsumu.sql"), "ibes_stats.parquet"),
        (os.path.join(migrations_dir, "007_ibes_actu.sql"),     "ibes_act.parquet"),
    ]

    need_extract = not reuse
    missing = []
    for _, fname in artifacts:
        if not os.path.isfile(os.path.join(outdir, fname)):
            missing.append(fname)
    if reuse and missing:
        print(f"[warn] Missing files detected: {missing}. Will extract missing pieces.")
        need_extract = True

    if need_extract:
        print("[info] Connecting to WRDS ...")
        conn = wrds.Connection(wrds_username=wrds_user, verbose=True)
        params = {"start": start, "end": end}

        for sqlfile, outfile in artifacts:
            outpath = os.path.join(outdir, outfile)
            if os.path.isfile(outpath):
                print(f"[skip] Already present: {outfile}")
                continue
            print(f"[extract] {sqlfile} -> {outfile}")
            query_to_parquet(conn, sqlfile, outpath, params=params, chunksize=chunksize)
            print(f"[ok] Saved {outfile}")

    # DuckDB join
    con = duckdb.connect(database=":memory:")
    views_from_parquet(con, "dsf",         os.path.join(outdir, "dsf.parquet"))
    views_from_parquet(con, "stocknames",  os.path.join(outdir, "stocknames.parquet"))
    views_from_parquet(con, "secm",        os.path.join(outdir, "secm.parquet"))
    views_from_parquet(con, "fundq",       os.path.join(outdir, "fundq.parquet"))
    views_from_parquet(con, "ff",          os.path.join(outdir, "ff.parquet"))
    views_from_parquet(con, "ibes_stats",  os.path.join(outdir, "ibes_stats.parquet"))
    views_from_parquet(con, "ibes_act",    os.path.join(outdir, "ibes_act.parquet"))

    if not os.path.isfile(join_sql):
        raise FileNotFoundError(f"Join SQL not found: {join_sql}")

    sql_join = open(join_sql).read()
    row_count = con.execute(sql_join).fetchone()[0]
    print(f"[result] Final joined row count: {row_count:,}")
    print(f"[done] Run folder: {outdir}")

    return {
        "row_count": row_count,
        "run_folder": outdir,
        "reuse": reuse
    }

res = wrds_extract_and_join(
    wrds_user="wboughattas",
    start="2010-01-01",
    end="2020-12-31",
    use_run="last"   # or "new" or "run_20250112_142233"
)
print(res)
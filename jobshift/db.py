"""DuckDB 連線。單檔資料庫，沒有 server，沒有密碼，沒有 docker service。"""
from __future__ import annotations

import duckdb

from jobshift import settings as S


def connect(read_only: bool = False) -> duckdb.DuckDBPyConnection:
    return duckdb.connect(str(S.DB_PATH), read_only=read_only)


def table_exists(con: duckdb.DuckDBPyConnection, name: str) -> bool:
    q = "SELECT count(*) FROM information_schema.tables WHERE table_name = ?"
    return con.execute(q, [name]).fetchone()[0] > 0


def replace_table(con: duckdb.DuckDBPyConnection, name: str, df) -> None:
    """整表覆寫。這個專案的表都是「重算即重建」，沒有增量更新的必要。"""
    con.register("_tmp_df", df)
    con.execute(f"CREATE OR REPLACE TABLE {name} AS SELECT * FROM _tmp_df")
    con.unregister("_tmp_df")


def upsert_snapshot(con: duckdb.DuckDBPyConnection, name: str, df, snapshot: str) -> None:
    """同一快照重跑時，先刪掉該快照的舊列再插入，避免重複。"""
    con.register("_tmp_df", df)
    if table_exists(con, name):
        con.execute(f"DELETE FROM {name} WHERE snapshot_date = ?", [snapshot])
        con.execute(f"INSERT INTO {name} SELECT * FROM _tmp_df")
    else:
        con.execute(f"CREATE TABLE {name} AS SELECT * FROM _tmp_df")
    con.unregister("_tmp_df")

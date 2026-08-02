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
    try:
        if table_exists(con, name):
            existing = [r[0] for r in con.execute(
                f"SELECT column_name FROM information_schema.columns "
                f"WHERE table_name = '{name}' ORDER BY ordinal_position").fetchall()]
            if existing != list(df.columns):
                # 欄位定義變了（例如新增分類維度）。硬 INSERT 只會拿到看不懂的
                # Binder Error，或更糟：欄位對錯位而不報錯。這裡明說該怎麼修。
                added = [c for c in df.columns if c not in existing]
                removed = [c for c in existing if c not in df.columns]
                raise SystemExit(
                    f"[db] {name} 表的欄位結構已經改變，不能直接插入。\n"
                    f"     新增：{added or '無'}　移除：{removed or '無'}\n"
                    f"     舊表 {len(existing)} 欄、新資料 {len(df.columns)} 欄。\n"
                    f"     修法：重建整張表 —— docker compose run --rm ingest "
                    f"python -m jobshift.ingest --all --rebuild\n"
                    f"     （所有快照都會從 data/raw 的原始檔重新清洗，不會遺失資料；"
                    f"重建後要重跑 analyze。）"
                )
            con.execute(f"DELETE FROM {name} WHERE snapshot_date = ?", [snapshot])
            con.execute(f"INSERT INTO {name} SELECT * FROM _tmp_df")
        else:
            con.execute(f"CREATE TABLE {name} AS SELECT * FROM _tmp_df")
    finally:
        con.unregister("_tmp_df")

"""FastAPI：這個專案唯一的資料出口。儀表板只呼叫這裡，不自己接 DuckDB —— 查詢邏輯只有一份。"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query

from jobshift import db
from jobshift import settings as S

app = FastAPI(title="jobshift API", version="1.0",
              description="職缺語意結構的跨期遷移分析")


def q(sql: str, params: list | None = None) -> list[dict]:
    with db.connect(read_only=True) as con:
        return con.execute(sql, params or []).df().to_dict(orient="records")


@lru_cache(maxsize=4)
def _vectors(snapshot: str):
    """記憶體映射載入，不整個吃進 RAM。"""
    vecs = np.load(S.vec_npy(snapshot), mmap_mode="r")
    ids = pd.read_parquet(S.vec_ids(snapshot))["job_id"].astype(str).tolist()
    return vecs, ids, {j: i for i, j in enumerate(ids)}


@app.get("/health")
def health():
    try:
        with db.connect(read_only=True) as con:
            tables = [r[0] for r in con.execute(
                "SELECT table_name FROM information_schema.tables").fetchall()]
        return {"status": "ok", "tables": sorted(tables)}
    except Exception as exc:
        return {"status": "degraded", "detail": str(exc)}


@app.get("/meta")
def meta():
    rows = q("SELECT key, value FROM analysis_meta")
    return {r["key"]: r["value"] for r in rows}


@app.get("/overview")
def overview():
    totals = q("SELECT snapshot_date, count(*) AS n, "
               "median(salary_month_min) AS median_salary "
               "FROM jobs GROUP BY 1 ORDER BY 1")
    industry = q("""
        WITH t AS (SELECT snapshot_date, industry_bucket, count(*) n FROM jobs GROUP BY 1,2)
        SELECT industry_bucket, snapshot_date, n,
               n::DOUBLE / sum(n) OVER (PARTITION BY snapshot_date) AS share
        FROM t ORDER BY industry_bucket, snapshot_date
    """)
    city = q("""
        WITH t AS (SELECT snapshot_date, city_name, count(*) n FROM jobs
                   WHERE city_name IS NOT NULL GROUP BY 1,2)
        SELECT city_name, snapshot_date, n,
               n::DOUBLE / sum(n) OVER (PARTITION BY snapshot_date) AS share
        FROM t QUALIFY sum(n) OVER (PARTITION BY city_name) > 300
        ORDER BY city_name, snapshot_date
    """)
    return {"totals": totals, "industry": industry, "city": city}


@app.get("/clusters")
def clusters(status: str | None = None, limit: int = Query(500, le=2000)):
    sql = "SELECT * FROM clusters"
    params: list = []
    if status:
        sql += " WHERE status = ?"
        params.append(status)
    sql += " ORDER BY abs(share_delta_pp) DESC LIMIT ?"
    params.append(limit)
    return q(sql, params)


@app.get("/clusters/{cluster_id}/jobs")
def cluster_jobs(cluster_id: int, snapshot: str | None = None,
                 limit: int = Query(50, le=500)):
    sql = """
        SELECT j.snapshot_date, j.job_id, j.job_title, j.company_name,
               j.industry_bucket, j.city_name, j.salary_desc, j.job_url, c.sim
        FROM job_cluster c JOIN jobs j
          ON j.job_id = c.job_id AND j.snapshot_date = c.snapshot_date
        WHERE c.cluster_id = ?
    """
    params: list = [cluster_id]
    if snapshot:
        sql += " AND c.snapshot_date = ?"
        params.append(snapshot)
    sql += " ORDER BY c.sim DESC LIMIT ?"
    params.append(limit)
    return q(sql, params)


@app.get("/flows")
def flows(snapshot: str | None = None, min_n: int = 20):
    sql = ("SELECT f.*, c.label, c.status FROM flow_industry_cluster f "
           "LEFT JOIN clusters c USING (cluster_id) WHERE f.n >= ?")
    params: list = [min_n]
    if snapshot:
        sql += " AND f.snapshot_date = ?"
        params.append(snapshot)
    return q(sql + " ORDER BY f.n DESC", params)


@app.get("/cluster-flow")
def cluster_flow(min_n: int = 5, include_self: bool = True):
    """遷移矩陣。from/to = -999 下架、-998 新進。"""
    sql = ("SELECT f.*, cf.label AS from_label, ct.label AS to_label "
           "FROM cluster_flow f "
           "LEFT JOIN clusters cf ON cf.cluster_id = f.from_cluster "
           "LEFT JOIN clusters ct ON ct.cluster_id = f.to_cluster "
           "WHERE f.n >= ?")
    if not include_self:
        sql += " AND f.kind <> 'stayed'"
    return q(sql + " ORDER BY f.n DESC", [min_n])


@app.get("/survival")
def survival():
    return q("SELECT * FROM survival ORDER BY churn_rate DESC")


@app.get("/company-shift")
def company_shift(limit: int = Query(100, le=1000), min_jobs: int = 3):
    return q("""
        SELECT s.*, cf.label AS from_label, ct.label AS to_label
        FROM company_shift s
        LEFT JOIN clusters cf ON cf.cluster_id = s.from_cluster
        LEFT JOIN clusters ct ON ct.cluster_id = s.to_cluster
        WHERE s.n_old >= ? AND s.n_new >= ?
        ORDER BY s.shift_score DESC LIMIT ?
    """, [min_jobs, min_jobs, limit])


@app.get("/similar/{job_id}")
def similar(job_id: str, snapshot: str | None = None, k: int = Query(10, le=50)):
    """用 bge-m3 向量找語意最近的職缺。跨期查詢就是「這個職缺兩個月前對應到什麼」。"""
    src_snap = snapshot or S.SNAPSHOT_DATE
    rows = q("SELECT snapshot_date FROM jobs WHERE job_id = ?", [job_id])
    if not rows:
        raise HTTPException(404, f"找不到 job_id={job_id}")
    home = rows[0]["snapshot_date"]
    vh, ids_h, idx_h = _vectors(home)
    if job_id not in idx_h:
        raise HTTPException(404, f"job_id={job_id} 沒有向量")
    v = np.asarray(vh[idx_h[job_id]], dtype=np.float32)

    vt, ids_t, _ = _vectors(src_snap)
    sims = np.asarray(vt, dtype=np.float32) @ v
    top = np.argsort(-sims)[: k + 1]
    hits = [{"job_id": ids_t[i], "similarity": float(sims[i])}
            for i in top if ids_t[i] != job_id][:k]
    detail = q(
        "SELECT job_id, job_title, company_name, industry_bucket, city_name, "
        "salary_desc, job_url FROM jobs WHERE snapshot_date = ? AND job_id IN "
        f"({','.join('?' * len(hits))})",
        [src_snap] + [h["job_id"] for h in hits]) if hits else []
    dm = {d["job_id"]: d for d in detail}
    return {"source": {"job_id": job_id, "snapshot_date": home},
            "target_snapshot": src_snap,
            "results": [{**dm.get(h["job_id"], {}), **h} for h in hits]}

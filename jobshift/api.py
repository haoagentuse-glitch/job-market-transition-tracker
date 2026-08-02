"""FastAPI：這個專案唯一的資料出口。儀表板只呼叫這裡，不自己接 DuckDB —— 查詢邏輯只有一份。

所有跨期端點都吃 `from_snapshot` / `to_snapshot`；不給就用時間軸上最後相鄰的兩期。
"""

from __future__ import annotations

import json
from functools import lru_cache

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query

from jobshift import db
from jobshift import settings as S

app = FastAPI(
    title="jobshift API",
    version="2.0",
    description="職缺語意結構的跨期遷移分析",
)


def q(sql: str, params: list | None = None) -> list[dict]:
    with db.connect(read_only=True) as con:
        return con.execute(sql, params or []).df().to_dict(orient="records")


def snapshots() -> list[str]:
    rows = q("SELECT value FROM analysis_meta WHERE key = 'snapshots'")
    if not rows:
        raise HTTPException(503, "分析尚未執行，請先跑 pipeline")
    return json.loads(rows[0]["value"])


def resolve_pair(frm: str | None, to: str | None) -> tuple[str, str]:
    """沒指定就用時間軸最後相鄰的兩期。"""
    snaps = snapshots()
    if len(snaps) < 2:
        raise HTTPException(503, "只有一份快照，還沒有跨期可比。再跑一次 pipeline。")
    return frm or snaps[-2], to or snaps[-1]


@lru_cache(maxsize=8)
def _vectors(snapshot: str):
    """記憶體映射載入，不整份吃進 RAM。"""
    vecs = np.load(S.vec_npy(snapshot), mmap_mode="r")
    ids = pd.read_parquet(S.vec_ids(snapshot))["job_id"].astype(str).tolist()
    return vecs, ids, {j: i for i, j in enumerate(ids)}


# ── 基本 ───────────────────────────────────────────────────────────
@app.get("/health")
def health():
    try:
        with db.connect(read_only=True) as con:
            tables = [
                r[0] for r in con.execute(
                    "SELECT table_name FROM information_schema.tables").fetchall()
            ]
        return {"status": "ok", "tables": sorted(tables)}
    except Exception as exc:
        return {"status": "degraded", "detail": str(exc)}


@app.get("/meta")
def meta():
    out = {r["key"]: r["value"] for r in q("SELECT key, value FROM analysis_meta")}
    out["snapshots"] = json.loads(out.get("snapshots", "[]"))
    return out


@app.get("/timeline")
def timeline():
    """每份快照的規模與薪資水準 —— 時間軸本身。"""
    return q("""
        SELECT snapshot_date, count(*) AS n,
               median(salary_month_min) AS median_salary,
               count(salary_month_min)::DOUBLE / count(*) AS salary_coverage,
               count(DISTINCT company_id) AS n_companies
        FROM jobs GROUP BY 1 ORDER BY 1
    """)


@app.get("/industry-series")
def industry_series():
    return q("""
        WITH t AS (SELECT snapshot_date, industry_bucket, count(*) n FROM jobs GROUP BY 1,2)
        SELECT industry_bucket, snapshot_date, n,
               n::DOUBLE / sum(n) OVER (PARTITION BY snapshot_date) AS share
        FROM t ORDER BY industry_bucket, snapshot_date
    """)


# ── 群集 ───────────────────────────────────────────────────────────
@app.get("/clusters")
def clusters(from_snapshot: str | None = None, to_snapshot: str | None = None,
             limit: int = Query(600, le=3000)):
    """群集目錄 + 指定兩期之間的消長。"""
    frm, to = resolve_pair(from_snapshot, to_snapshot)
    return q("""
        WITH a AS (SELECT cluster_id, n, share FROM cluster_timeseries WHERE snapshot_date = ?),
             b AS (SELECT cluster_id, n, share FROM cluster_timeseries WHERE snapshot_date = ?)
        SELECT c.*,
               COALESCE(a.n, 0) AS n_from, COALESCE(b.n, 0) AS n_to,
               COALESCE(b.n, 0) - COALESCE(a.n, 0) AS delta,
               COALESCE(a.share, 0) AS share_from, COALESCE(b.share, 0) AS share_to,
               (COALESCE(b.share, 0) - COALESCE(a.share, 0)) * 100 AS share_delta_pp,
               CASE WHEN COALESCE(a.n, 0) = 0 THEN NULL
                    ELSE (COALESCE(b.n, 0) - a.n) * 100.0 / a.n END AS growth_pct,
               CASE WHEN c.first_seen > ?              THEN '新生'
                    WHEN COALESCE(b.n, 0) = 0          THEN '消亡'
                    WHEN (COALESCE(b.share,0) - COALESCE(a.share,0)) * 100 >  0.2 THEN '擴張'
                    WHEN (COALESCE(b.share,0) - COALESCE(a.share,0)) * 100 < -0.2 THEN '萎縮'
                    ELSE '持平' END AS status
        FROM clusters c
        LEFT JOIN a USING (cluster_id)
        LEFT JOIN b USING (cluster_id)
        ORDER BY abs(COALESCE(b.share, 0) - COALESCE(a.share, 0)) DESC
        LIMIT ?
    """, [frm, to, frm, limit])


@app.get("/clusters/series")
def cluster_series(cluster_id: int | None = None, top: int = Query(15, le=60)):
    """群集佔比的完整時間序列。不指定 cluster_id 就取變化幅度最大的前 N 個。"""
    if cluster_id is not None:
        return q("""
            SELECT t.*, c.label FROM cluster_timeseries t JOIN clusters c USING (cluster_id)
            WHERE t.cluster_id = ? ORDER BY t.snapshot_date
        """, [cluster_id])
    return q("""
        WITH rng AS (
            SELECT cluster_id, max(share) - min(share) AS spread
            FROM cluster_timeseries WHERE cluster_id >= 0 GROUP BY 1
            ORDER BY spread DESC LIMIT ?
        )
        SELECT t.*, c.label FROM cluster_timeseries t
        JOIN rng USING (cluster_id) JOIN clusters c USING (cluster_id)
        ORDER BY c.label, t.snapshot_date
    """, [top])


@app.get("/clusters/{cluster_id}/jobs")
def cluster_jobs(cluster_id: int, snapshot: str | None = None,
                 limit: int = Query(50, le=500)):
    sql = """
        SELECT j.snapshot_date, j.job_id, j.job_title, j.company_name,
               j.industry_bucket, j.city_name, j.salary_desc, j.job_url, c.sim
        FROM job_cluster c
        JOIN jobs j ON j.job_id = c.job_id AND j.snapshot_date = c.snapshot_date
        WHERE c.cluster_id = ?
    """
    params: list = [cluster_id]
    if snapshot:
        sql += " AND c.snapshot_date = ?"
        params.append(snapshot)
    sql += " ORDER BY c.sim DESC LIMIT ?"
    params.append(limit)
    return q(sql, params)


# ── 跨期遷移 ───────────────────────────────────────────────────────
@app.get("/cluster-flow")
def cluster_flow(from_snapshot: str | None = None, to_snapshot: str | None = None,
                 min_n: int = 5, include_self: bool = True):
    """遷移矩陣。from_cluster = -998 新進、to_cluster = -999 下架。"""
    frm, to = resolve_pair(from_snapshot, to_snapshot)
    sql = """
        SELECT f.*, cf.label AS from_label, ct.label AS to_label
        FROM cluster_flow f
        LEFT JOIN clusters cf ON cf.cluster_id = f.from_cluster
        LEFT JOIN clusters ct ON ct.cluster_id = f.to_cluster
        WHERE f.from_snapshot = ? AND f.to_snapshot = ? AND f.n >= ?
    """
    if not include_self:
        sql += " AND f.kind <> 'stayed'"
    return q(sql + " ORDER BY f.n DESC", [frm, to, min_n])


@app.get("/survival")
def survival(from_snapshot: str | None = None, to_snapshot: str | None = None):
    frm, to = resolve_pair(from_snapshot, to_snapshot)
    return q("SELECT * FROM survival WHERE from_snapshot = ? AND to_snapshot = ? "
             "ORDER BY churn_rate DESC", [frm, to])


@app.get("/company-shift")
def company_shift(from_snapshot: str | None = None, to_snapshot: str | None = None,
                  limit: int = Query(200, le=2000), min_jobs: int = 3):
    frm, to = resolve_pair(from_snapshot, to_snapshot)
    return q("""
        SELECT s.*, cf.label AS from_label, ct.label AS to_label
        FROM company_shift s
        LEFT JOIN clusters cf ON cf.cluster_id = s.from_cluster
        LEFT JOIN clusters ct ON ct.cluster_id = s.to_cluster
        WHERE s.from_snapshot = ? AND s.to_snapshot = ?
          AND s.n_from >= ? AND s.n_to >= ?
        ORDER BY s.shift_score DESC LIMIT ?
    """, [frm, to, min_jobs, min_jobs, limit])


@app.get("/industry-composition")
def industry_composition(snapshot: str | None = None, min_n: int = 20):
    sql = ("SELECT f.*, c.label FROM flow_industry_cluster f "
           "LEFT JOIN clusters c USING (cluster_id) WHERE f.n >= ?")
    params: list = [min_n]
    if snapshot:
        sql += " AND f.snapshot_date = ?"
        params.append(snapshot)
    return q(sql + " ORDER BY f.n DESC", params)


# ── 向量檢索 ───────────────────────────────────────────────────────
@app.get("/similar/{job_id}")
def similar(job_id: str, snapshot: str | None = None, k: int = Query(10, le=50)):
    """用 bge-m3 向量找語意最近的職缺。把目標快照切到前一期，
    就是在問「這個職缺在兩個月前對應到什麼」。"""
    rows = q("SELECT snapshot_date FROM jobs WHERE job_id = ? ORDER BY snapshot_date DESC",
             [job_id])
    if not rows:
        raise HTTPException(404, f"找不到 job_id={job_id}")
    home = rows[0]["snapshot_date"]
    target = snapshot or home

    _, _, idx_home = _vectors(home)
    if job_id not in idx_home:
        raise HTTPException(404, f"job_id={job_id} 沒有向量")
    v = np.asarray(_vectors(home)[0][idx_home[job_id]], dtype=np.float32)

    vt, ids_t, _ = _vectors(target)
    sims = np.asarray(vt, dtype=np.float32) @ v
    order = np.argsort(-sims)[: k + 1]
    hits = [{"job_id": ids_t[i], "similarity": float(sims[i])}
            for i in order if ids_t[i] != job_id][:k]
    if not hits:
        return {"source": {"job_id": job_id, "snapshot_date": home},
                "target_snapshot": target, "results": []}

    detail = q(
        "SELECT job_id, job_title, company_name, industry_bucket, city_name, "
        f"salary_desc, job_url FROM jobs WHERE snapshot_date = ? AND job_id IN "
        f"({','.join('?' * len(hits))})",
        [target] + [h["job_id"] for h in hits])
    dm = {d["job_id"]: d for d in detail}
    return {"source": {"job_id": job_id, "snapshot_date": home},
            "target_snapshot": target,
            "results": [{**dm.get(h["job_id"], {}), **h} for h in hits]}

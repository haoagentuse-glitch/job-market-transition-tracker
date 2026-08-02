"""把所有視圖匯出成靜態 JSON，供 GitHub Pages 版使用。

單一快照的資料不會變，所有聚合結果都是定值 —— 因此不需要伺服器。
本腳本直接在行程內呼叫 api 模組的函式，與線上版走同一套查詢邏輯，
不另外寫一份 SQL，避免兩版數字對不起來。
"""

from __future__ import annotations

import json
from pathlib import Path

from jobshift import api, taxonomy
from jobshift import settings as S

OUT = Path(__file__).resolve().parent.parent / "docs" / "data"
CLUSTER_JOB_LIMIT = 30
CONCEPT_JOB_LIMIT = 100


def build() -> tuple[dict, dict]:
    meta = api.meta()
    cmeta = api.concept_meta()
    # 所有帶 Query() 預設值的參數都要明確給值：直接呼叫函式時 FastAPI 不會介入解析，
    # 預設值仍是 Query 物件，傳進 SQL 會炸在型別轉換。
    clusters = api.clusters(limit=3000)
    concepts = api.concept_list()

    core = {
        "meta": meta,
        "concept_meta": cmeta,
        "timeline": api.timeline(),
        "clusters": clusters,
        "concepts": concepts,
        "concept_salary": api.concept_salary(),
        "concept_examples": api.concept_examples(),
        "concept_candidates": api.concept_candidates(),
        "dimensions": api.dimensions(),
        "cross": {d: api.concept_cross(dimension=d, min_n=5) for d in taxonomy.DIMENSIONS},
        "industry_series": {d: api.industry_series(dimension=d) for d in taxonomy.DIMENSIONS},
    }

    jobs = {
        "by_cluster": {
            str(c["cluster_id"]): api.cluster_jobs(c["cluster_id"], limit=CLUSTER_JOB_LIMIT)
            for c in clusters
        },
        "by_concept": {
            c["concept"]: api.concept_jobs(c["concept"], limit=CONCEPT_JOB_LIMIT)
            for c in concepts
        },
    }
    return core, jobs


def write(name: str, payload) -> int:
    path = OUT / name
    text = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    path.write_text(text, encoding="utf-8")
    kb = len(text.encode("utf-8")) / 1024
    print(f"  {name:20s} {kb:>9,.0f} KB")
    return len(text.encode("utf-8"))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"[export] 快照 {S.REFERENCE_SNAPSHOT} → {OUT}")
    core, jobs = build()
    total = write("core.json", core) + write("jobs.json", jobs)
    print(f"  {'合計':20s} {total / 1024 / 1024:>9,.2f} MB")
    print(f"[export] 群集 {len(core['clusters'])}｜概念 {len(core['concepts'])}"
          f"｜職缺清單 {sum(len(v) for v in jobs['by_cluster'].values()):,} 列")


if __name__ == "__main__":
    main()

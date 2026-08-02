"""技能偵測。以精修過的關鍵字樣式認定，向量錨點僅作佐證與候選擴充。

方法選擇的依據見 concepts.py 檔頭：向量錨點分類在本資料上實測分離度不足
（正例中位數低於全體 p99），因此不作為認定依據。

產出四張表：
  concept_scores    命中職缺 × 概念，含錨點相似度
  concept_summary   各概念的規模、薪資、描述長度，以及錨點相似度的分離度指標
  concept_examples  代表職缺
  concept_candidates 關鍵字未命中但錨點相似度最高者，標記為待驗證，不計入統計
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from jobshift import concepts as C
from jobshift import db
from jobshift import settings as S


def load(snapshot: str) -> tuple[pd.DataFrame, np.ndarray]:
    vecs = np.load(S.vec_npy(snapshot))
    ids = pd.read_parquet(S.vec_ids(snapshot))["job_id"].astype(str)
    with db.connect(read_only=True) as con:
        meta = con.execute(
            "SELECT job_id, job_title, company_name, bucket_label, industry_class, "
            "city_name, salary_month_min, description, job_url "
            "FROM jobs WHERE snapshot_date = ?", [snapshot]).df()
    meta["job_id"] = meta["job_id"].astype(str)
    meta = ids.to_frame().merge(meta, on="job_id", how="left")
    if len(meta) != len(vecs):
        raise SystemExit(f"metadata {len(meta)} != 向量 {len(vecs)}")
    return meta, vecs


def anchor_vectors() -> tuple[list[str], np.ndarray]:
    import torch
    from sentence_transformers import SentenceTransformer

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(S.EMBED_MODEL, device=dev)
    model.max_seq_length = S.EMBED_MAX_LEN
    if dev == "cuda":
        model = model.half()

    names, mats = [], []
    for name, sents in C.ANCHORS.items():
        v = model.encode(sents, normalize_embeddings=True, convert_to_numpy=True,
                         show_progress_bar=False).astype(np.float32)
        c = v.mean(axis=0)
        mats.append(c / max(np.linalg.norm(c), 1e-9))
        names.append(name)
    return names, np.vstack(mats)


def main() -> None:
    snap = S.REFERENCE_SNAPSHOT
    meta, vecs = load(snap)
    n = len(meta)
    print(f"[skills] {snap}：{n:,} 筆")

    names, anchors = anchor_vectors()
    sims = vecs @ anchors.T
    text = (meta.job_title.fillna("") + " " + meta.description.fillna("")).to_numpy()
    kw = C.compiled_keywords()
    base_salary = meta.salary_month_min.dropna()

    rows, summary, examples, candidates = [], [], [], []
    for i, name in enumerate(names):
        s = sims[:, i]
        hit = np.fromiter((bool(kw[name].search(t)) for t in text), bool, n)
        idx = np.where(hit)[0]

        sal = meta.salary_month_min[hit].dropna()
        # 分離度：正例相似度中位數距全體平均幾個標準差。用來說明為何不採信向量認定。
        sep = (float(np.median(s[hit])) - float(s.mean())) / (float(s.std()) + 1e-9) \
            if hit.any() else np.nan
        summary.append({
            "concept": name,
            "is_ai": name in C.AI_CONCEPTS,
            "n": int(hit.sum()),
            "share": float(hit.mean()),
            "median_salary": float(sal.median()) if len(sal) else np.nan,
            "salary_premium": (float(sal.median() / base_salary.median())
                               if len(sal) and len(base_salary) else np.nan),
            "salary_coverage": float(len(sal) / max(hit.sum(), 1)),
            "median_desc_len": (float(meta.description.str.len()[hit].median())
                                if hit.any() else np.nan),
            "anchor_sim_median": float(np.median(s[hit])) if hit.any() else np.nan,
            "anchor_sim_p99_all": float(np.percentile(s, 99)),
            "separation_sigma": sep,
        })

        for j in idx[np.argsort(-s[idx])][:10]:
            examples.append({"concept": name, "job_title": meta.job_title[j],
                             "company_name": meta.company_name[j],
                             "bucket_label": meta.bucket_label[j],
                             "industry_class": meta.industry_class[j],
                             "city_name": meta.city_name[j],
                             "salary_month_min": meta.salary_month_min[j],
                             "anchor_similarity": float(s[j]),
                             "job_url": meta.job_url[j]})

        miss = np.where(~hit)[0]
        for j in miss[np.argsort(-s[miss])][:C.CANDIDATE_TOP_N]:
            candidates.append({"concept": name, "job_title": meta.job_title[j],
                               "company_name": meta.company_name[j],
                               "bucket_label": meta.bucket_label[j],
                               "anchor_similarity": float(s[j]),
                               "job_url": meta.job_url[j]})

        rows.append(pd.DataFrame({
            "job_id": meta.job_id[hit].to_numpy(),
            "concept": name,
            "anchor_similarity": s[hit].astype(np.float32),
        }))
        print(f"  {name:16s} {hit.sum():>5,} 筆（{hit.mean():>5.2%}）"
              f"｜月薪中位 {sal.median() if len(sal) else float('nan'):>8,.0f}"
              f"｜分離度 {sep:>5.2f}σ")

    scores = pd.concat(rows, ignore_index=True)
    ai_ids = set(scores[scores.concept.isin(C.AI_CONCEPTS)].job_id)
    digital_ids = set(scores.job_id)
    print(f"\n[skills] AI 技能職缺 {len(ai_ids):,}（{len(ai_ids) / n:.2%}）"
          f"｜數位技能職缺 {len(digital_ids):,}（{len(digital_ids) / n:.2%}）")

    with db.connect() as con:
        db.replace_table(con, "concept_scores", scores)
        db.replace_table(con, "concept_summary", pd.DataFrame(summary))
        db.replace_table(con, "concept_examples", pd.DataFrame(examples))
        db.replace_table(con, "concept_candidates", pd.DataFrame(candidates))
        db.replace_table(con, "concept_meta", pd.DataFrame([
            {"key": "snapshot", "value": snap},
            {"key": "n_jobs", "value": str(n)},
            {"key": "method", "value": "keyword"},
            {"key": "ai_concepts", "value": "、".join(C.AI_CONCEPTS)},
            {"key": "n_ai_jobs", "value": str(len(ai_ids))},
            {"key": "n_digital_jobs", "value": str(len(digital_ids))},
        ]))
    print("[skills] 完成")


if __name__ == "__main__":
    main()

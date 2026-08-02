"""分群、跨期指派、遷移。整條分析的唯一入口，結果全部落 DuckDB。

方法（三個刻意的選擇，各自有替代方案被否決）：

1. 參考座標系建在「舊快照」上。分群只 fit 一次，新快照是「被指派進來」的。
   否決的替代：兩期各自分群再對齊 —— 群集編號會漂移，遷移矩陣會變成對齊演算法的產物。

2. 跨期指派用「對舊快照的 kNN 多數決」，不用 HDBSCAN 的 approximate_predict。
   HDBSCAN 的群在 UMAP 空間常是非凸的，用群心距離指派會錯得很難察覺；
   kNN 直接在原始 1024 維 cosine 空間投票，非凸也不怕，而且票數與相似度天然是信心指標。

3. 「遷移」做三層。同一則職缺的描述兩期不會變，硬做 job 層級的群間移動是假的：
   ① 產業 × 群集的交叉結構重組  ② 群集消長／新生／消亡  ③ 同一家公司招募重心的位移。
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from sklearn.cluster import HDBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score

from jobshift import db
from jobshift import settings as S

EMERGENT_OFFSET = 1000      # 新生群集的 id 從 1000 起跳，跟舊參考群集分得開


# ── 載入 ───────────────────────────────────────────────────────────
def load(snapshot: str) -> tuple[pd.DataFrame, np.ndarray]:
    vecs = np.load(S.vec_npy(snapshot))
    ids = pd.read_parquet(S.vec_ids(snapshot))["job_id"].astype(str)
    with db.connect(read_only=True) as con:
        meta = con.execute(
            "SELECT job_id, job_title, company_id, company_name, industry_bucket, "
            "industry_name, city_name, salary_month_min, job_url "
            "FROM jobs WHERE snapshot_date = ?", [snapshot]
        ).df()
    meta["job_id"] = meta["job_id"].astype(str)
    meta = ids.to_frame().merge(meta, on="job_id", how="left")
    assert len(meta) == len(vecs), f"{snapshot}: metadata {len(meta)} != 向量 {len(vecs)}"
    return meta, vecs


# ── 參考座標系：在舊快照上分群 ──────────────────────────────────────
def fit_reference(vecs: np.ndarray) -> tuple[np.ndarray, dict, PCA, object]:
    n = len(vecs)
    print(f"[analyze] 建參考座標系：{n} 筆 × {vecs.shape[1]} 維")

    pca = PCA(n_components=min(S.PCA_DIM, vecs.shape[1]), random_state=0).fit(vecs)
    low = pca.transform(vecs)
    print(f"[analyze] PCA {vecs.shape[1]}→{low.shape[1]} "
          f"（保留變異 {pca.explained_variance_ratio_.sum():.1%}）")

    import umap
    reducer = umap.UMAP(
        n_components=S.UMAP_DIM, n_neighbors=S.UMAP_NEIGHBORS,
        min_dist=0.0, metric="cosine", random_state=42, verbose=True,
    ).fit(low)
    emb = reducer.embedding_

    min_size = max(25, int(n * S.HDBSCAN_MIN_FRAC))
    labels = HDBSCAN(min_cluster_size=min_size, min_samples=10,
                     cluster_selection_method="eom").fit_predict(emb)
    noise = float((labels == -1).mean())
    n_clu = len(set(labels)) - (1 if -1 in labels else 0)
    print(f"[analyze] HDBSCAN(min_cluster_size={min_size}) → {n_clu} 群，噪點 {noise:.1%}")

    method = f"hdbscan(min_cluster_size={min_size})"
    if noise > S.NOISE_FALLBACK or n_clu < 8:
        print(f"[analyze] 噪點超過 {S.NOISE_FALLBACK:.0%} 或群數過少 → 退回 KMeans")
        sub = emb[np.random.default_rng(0).choice(n, min(8000, n), replace=False)]
        best_k, best_s = None, -1.0
        for k in S.KMEANS_K_GRID:
            lab = KMeans(n_clusters=k, n_init=4, random_state=0).fit_predict(sub)
            s = silhouette_score(sub, lab)
            print(f"           k={k:>3} silhouette={s:.4f}")
            if s > best_s:
                best_k, best_s = k, s
        labels = KMeans(n_clusters=best_k, n_init=10, random_state=0).fit_predict(emb)
        method = f"kmeans(k={best_k}, silhouette={best_s:.4f})"
        print(f"[analyze] 採用 {method}")

    info = {"method": method, "noise_ratio": noise, "n_clusters": int(
        len(set(labels)) - (1 if -1 in labels else 0))}
    return labels, info, pca, reducer


# ── 群集命名：不用斷詞，全靠 embedding + 字元 n-gram ─────────────────
def label_clusters(meta: pd.DataFrame, vecs: np.ndarray, labels: np.ndarray) -> pd.DataFrame:
    titles = meta["job_title"].fillna("").tolist()
    tfidf = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4),
                            min_df=max(2, len(titles) // 5000), max_features=60000)
    X = tfidf.fit_transform(titles)
    vocab = np.array(tfidf.get_feature_names_out())
    global_mean = np.asarray(X.mean(axis=0)).ravel()

    rows = []
    for cid in sorted(set(labels)):
        if cid == -1:
            continue
        mask = labels == cid
        # 代表職稱：離群心最近的真實職稱（比詞袋好讀，也不需要 tokenizer）
        center = vecs[mask].mean(axis=0)
        center /= max(np.linalg.norm(center), 1e-9)
        sims = vecs[mask] @ center
        top_idx = np.argsort(-sims)[:8]
        member_titles = pd.Series(np.array(titles)[mask][top_idx]).drop_duplicates()

        # 鑑別詞：該群 tfidf 均值減全域均值，取最突出的字元 n-gram
        c_mean = np.asarray(X[mask].mean(axis=0)).ravel()
        terms = vocab[np.argsort(-(c_mean - global_mean))[:12]]
        terms = [t.strip() for t in terms if len(t.strip()) >= 2][:6]

        rows.append({
            "cluster_id": int(cid),
            "kind": "reference",
            "label": member_titles.iloc[0] if len(member_titles) else f"群 {cid}",
            "top_titles": " / ".join(member_titles.head(5)),
            "top_terms": " ".join(terms),
        })
    return pd.DataFrame(rows)


# ── 跨期指派：對舊快照做 kNN 多數決 ─────────────────────────────────
def assign_by_knn(new_vecs: np.ndarray, old_vecs: np.ndarray,
                  old_labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    try:
        import torch
        dev = "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        torch, dev = None, "cpu"

    k = S.KNN_K
    n = len(new_vecs)
    out_lab = np.full(n, -1, dtype=np.int64)
    out_sim = np.zeros(n, dtype=np.float32)
    chunk = 1024
    print(f"[analyze] kNN 指派 {n} 筆 → {len(old_vecs)} 筆參考（k={k}, device={dev}）")

    if torch is not None:
        old_t = torch.from_numpy(old_vecs).to(dev)
    for start in range(0, n, chunk):
        blk = new_vecs[start:start + chunk]
        if torch is not None:
            sim = (torch.from_numpy(blk).to(dev) @ old_t.T)
            top_sim, top_idx = torch.topk(sim, k, dim=1)
            top_sim = top_sim.cpu().numpy()
            top_idx = top_idx.cpu().numpy()
        else:
            sim = blk @ old_vecs.T
            top_idx = np.argpartition(-sim, k, axis=1)[:, :k]
            top_sim = np.take_along_axis(sim, top_idx, axis=1)

        neigh_lab = old_labels[top_idx]
        for i in range(len(blk)):
            lab, sm = neigh_lab[i], top_sim[i]
            valid = lab != -1
            if not valid.any():
                continue
            vals, counts = np.unique(lab[valid], return_counts=True)
            win = vals[counts.argmax()]
            ratio = counts.max() / k
            mean_sim = float(sm[lab == win].mean())
            if ratio >= S.KNN_VOTE_RATIO and mean_sim >= S.KNN_SIM_FLOOR:
                out_lab[start + i] = win
                out_sim[start + i] = mean_sim
            else:
                out_sim[start + i] = mean_sim
    hit = (out_lab != -1).mean()
    print(f"[analyze] 指派成功率 {hit:.1%}，未歸類 {(out_lab == -1).sum()} 筆（新型態候選）")
    return out_lab, out_sim


def find_emergent(new_meta: pd.DataFrame, new_vecs: np.ndarray, labels: np.ndarray,
                  pca: PCA, reducer) -> tuple[np.ndarray, pd.DataFrame]:
    """未歸類的新職缺自己再分一次群 —— 這些就是舊快照裡沒有的形態。"""
    idx = np.where(labels == -1)[0]
    if len(idx) < 60:
        print("[analyze] 未歸類數量太少，不做新生群集偵測")
        return labels, pd.DataFrame(columns=["cluster_id", "kind", "label",
                                             "top_titles", "top_terms"])
    emb = reducer.transform(pca.transform(new_vecs[idx]))
    min_size = max(25, int(len(idx) * 0.02))
    sub = HDBSCAN(min_cluster_size=min_size, min_samples=5).fit_predict(emb)
    n_new = len(set(sub)) - (1 if -1 in sub else 0)
    print(f"[analyze] 新生群集偵測：{len(idx)} 筆未歸類 → {n_new} 個新群")
    if n_new == 0:
        return labels, pd.DataFrame(columns=["cluster_id", "kind", "label",
                                             "top_titles", "top_terms"])

    labels = labels.copy()
    labels[idx] = np.where(sub == -1, -1, sub + EMERGENT_OFFSET)
    rows = label_clusters(new_meta.iloc[idx].reset_index(drop=True),
                          new_vecs[idx], np.where(sub == -1, -1, sub))
    rows["cluster_id"] += EMERGENT_OFFSET
    rows["kind"] = "emergent"
    return labels, rows


# ── 三層遷移 ───────────────────────────────────────────────────────
def build_outputs(old_meta, new_meta, old_lab, new_lab, new_sim,
                  clusters: pd.DataFrame, old_vecs, new_vecs, info: dict) -> dict:
    old_meta = old_meta.assign(cluster_id=old_lab, sim=1.0)
    new_meta = new_meta.assign(cluster_id=new_lab, sim=new_sim)

    # ② 群集消長
    c_old = old_meta.cluster_id.value_counts()
    c_new = new_meta.cluster_id.value_counts()
    n_old_tot, n_new_tot = len(old_meta), len(new_meta)
    clusters = clusters.set_index("cluster_id")
    clusters["n_old"] = c_old.reindex(clusters.index).fillna(0).astype(int)
    clusters["n_new"] = c_new.reindex(clusters.index).fillna(0).astype(int)
    clusters["delta"] = clusters.n_new - clusters.n_old
    clusters["share_old"] = clusters.n_old / n_old_tot
    clusters["share_new"] = clusters.n_new / n_new_tot
    clusters["share_delta_pp"] = (clusters.share_new - clusters.share_old) * 100
    clusters["growth_pct"] = np.where(
        clusters.n_old > 0, (clusters.n_new - clusters.n_old) / clusters.n_old * 100, np.nan)
    clusters["status"] = np.select(
        [clusters.kind.eq("emergent"), clusters.n_new == 0,
         clusters.share_delta_pp > 0.2, clusters.share_delta_pp < -0.2],
        ["新生", "消亡", "擴張", "萎縮"], default="持平")
    for snap, m in (("old", old_meta), ("new", new_meta)):
        med = m.groupby("cluster_id").salary_month_min.median()
        clusters[f"salary_{snap}"] = med.reindex(clusters.index)
    clusters = clusters.reset_index()

    # ① 產業 × 群集 交叉結構
    flows = []
    for snap, m in ((S.BASE_SNAPSHOT, old_meta), (info["new_snapshot"], new_meta)):
        g = (m[m.cluster_id != -1]
             .groupby(["industry_bucket", "cluster_id"]).size()
             .reset_index(name="n"))
        g["snapshot_date"] = snap
        g["share_in_industry"] = g.n / g.groupby("industry_bucket").n.transform("sum")
        flows.append(g)
    flow_df = pd.concat(flows, ignore_index=True)

    # 存活／汰換
    old_ids, new_ids = set(old_meta.job_id), set(new_meta.job_id)
    surv = []
    for bucket in sorted(set(old_meta.industry_bucket) | set(new_meta.industry_bucket)):
        o = set(old_meta.loc[old_meta.industry_bucket == bucket, "job_id"])
        nn = set(new_meta.loc[new_meta.industry_bucket == bucket, "job_id"])
        keep = len(o & nn)
        surv.append({"industry_bucket": bucket, "n_old": len(o), "n_new": len(nn),
                     "n_survived": keep, "n_gone": len(o) - keep, "n_added": len(nn) - keep,
                     "churn_rate": (len(o) - keep) / len(o) if o else np.nan,
                     "renewal_rate": (len(nn) - keep) / len(nn) if nn else np.nan})
    surv_df = pd.DataFrame(surv)
    print(f"[analyze] 全體存活：{len(old_ids & new_ids)} / {len(old_ids)} "
          f"（{len(old_ids & new_ids) / len(old_ids):.1%}）")

    # ③ 公司層級語意位移：兩期都招募 ≥3 筆的公司，比較招募重心向量
    shift = []
    old_idx = {c: g.index.values for c, g in old_meta.groupby("company_id")}
    for comp, g_new in new_meta.groupby("company_id"):
        g_old_idx = old_idx.get(comp)
        if g_old_idx is None or len(g_old_idx) < 3 or len(g_new) < 3:
            continue
        vo = old_vecs[g_old_idx].mean(axis=0)
        vn = new_vecs[g_new.index.values].mean(axis=0)
        cos = float(vo @ vn / max(np.linalg.norm(vo) * np.linalg.norm(vn), 1e-9))
        fo = old_meta.loc[g_old_idx].cluster_id
        fo = fo[fo != -1]
        fn = g_new.cluster_id[g_new.cluster_id != -1]
        shift.append({
            "company_id": comp,
            "company_name": g_new.company_name.iloc[0],
            "industry_bucket": g_new.industry_bucket.iloc[0],
            "n_old": len(g_old_idx), "n_new": len(g_new),
            "cos_similarity": cos, "shift_score": 1 - cos,
            "from_cluster": int(fo.mode().iloc[0]) if len(fo) else -1,
            "to_cluster": int(fn.mode().iloc[0]) if len(fn) else -1,
        })
    shift_df = pd.DataFrame(shift).sort_values("shift_score", ascending=False)
    print(f"[analyze] 公司遷移：{len(shift_df)} 家兩期都有 ≥3 筆招募")

    # 遷移矩陣：存活職缺的 舊群集 → 新群集（多數是自環，離開自環的就是真的被重新定位），
    # 外加「下架」與「新進」兩個端點，讓 Sankey 的流量兩邊守恆。
    surv_ids = old_ids & new_ids
    surv_list = sorted(surv_ids)        # 固定順序，兩次 reindex 才對得起來
    o_c = old_meta.set_index("job_id").cluster_id
    n_c = new_meta.set_index("job_id").cluster_id
    pair = pd.DataFrame({"from_cluster": o_c.reindex(surv_list).to_numpy(),
                         "to_cluster": n_c.reindex(surv_list).to_numpy()})
    mat = pair.groupby(["from_cluster", "to_cluster"]).size().reset_index(name="n")
    mat["kind"] = np.where(mat.from_cluster == mat.to_cluster, "stayed", "moved")

    gone = (old_meta[~old_meta.job_id.isin(surv_ids)]
            .groupby("cluster_id").size().reset_index(name="n")
            .rename(columns={"cluster_id": "from_cluster"}))
    gone["to_cluster"], gone["kind"] = -999, "gone"          # -999 = 下架
    added = (new_meta[~new_meta.job_id.isin(surv_ids)]
             .groupby("cluster_id").size().reset_index(name="n")
             .rename(columns={"cluster_id": "to_cluster"}))
    added["from_cluster"], added["kind"] = -998, "added"     # -998 = 新進
    flow_mat = pd.concat([mat, gone, added], ignore_index=True)[
        ["from_cluster", "to_cluster", "n", "kind"]]
    moved = flow_mat.loc[flow_mat.kind == "moved", "n"].sum()
    print(f"[analyze] 遷移矩陣：存活 {len(surv_ids)} 筆，其中 {moved} 筆換了群集")

    jc = pd.concat([
        old_meta.assign(snapshot_date=S.BASE_SNAPSHOT),
        new_meta.assign(snapshot_date=info["new_snapshot"]),
    ])[["snapshot_date", "job_id", "cluster_id", "sim"]]

    return {"clusters": clusters, "job_cluster": jc, "flow_industry_cluster": flow_df,
            "cluster_flow": flow_mat, "survival": surv_df, "company_shift": shift_df}


# ── 主流程 ─────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", default=S.SNAPSHOT_DATE, help="新快照日期")
    args = ap.parse_args()

    old_meta, old_vecs = load(S.BASE_SNAPSHOT)
    new_meta, new_vecs = load(args.snapshot)

    old_lab, info, pca, reducer = fit_reference(old_vecs)
    info["new_snapshot"] = args.snapshot
    clusters = label_clusters(old_meta, old_vecs, old_lab)

    new_lab, new_sim = assign_by_knn(new_vecs, old_vecs, old_lab)
    new_lab, emergent = find_emergent(new_meta, new_vecs, new_lab, pca, reducer)
    if len(emergent):
        clusters = pd.concat([clusters, emergent], ignore_index=True)

    out = build_outputs(old_meta, new_meta, old_lab, new_lab, new_sim,
                        clusters, old_vecs, new_vecs, info)

    meta_rows = pd.DataFrame([
        {"key": "base_snapshot", "value": S.BASE_SNAPSHOT},
        {"key": "new_snapshot", "value": args.snapshot},
        {"key": "cluster_method", "value": info["method"]},
        {"key": "noise_ratio", "value": f"{info['noise_ratio']:.4f}"},
        {"key": "n_reference_clusters", "value": str(info["n_clusters"])},
        {"key": "params", "value": json.dumps(
            {"knn_k": S.KNN_K, "vote_ratio": S.KNN_VOTE_RATIO,
             "sim_floor": S.KNN_SIM_FLOOR, "umap_dim": S.UMAP_DIM}, ensure_ascii=False)},
    ])

    with db.connect() as con:
        for name, df in out.items():
            db.replace_table(con, name, df)
        db.replace_table(con, "analysis_meta", meta_rows)
    print("[analyze] 完成，已寫入 DuckDB："
          + "、".join(out.keys()) + "、analysis_meta")


if __name__ == "__main__":
    main()

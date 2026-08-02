"""分群、跨期指派、遷移。整條分析的唯一入口，結果全部落 DuckDB。

設計成「可以一直加快照」，不是一次性的兩期對照：

1. **參考座標系只建一次**，建在最早的那份快照上（`settings.REFERENCE_SNAPSHOT`）。
   之後每一份新快照都是「被指派進這個座標系」，所以群集編號跨期穩定，
   時間序列才有意義。否決的替代：每期各自分群再對齊 —— 編號會漂移，
   看到的變化會變成對齊演算法的產物而不是市場的變化。

2. **座標系會生長。** 某期出現一批指派不進去的職缺（＝舊快照裡沒有的形態），
   會被獨立分群、賦予新的群集編號，然後**併回參考池**。
   下一期就能直接比對到它們 —— 新型態一旦出現就被納入詞彙表，不會每期重新發明。

3. **跨期指派用對參考池的 kNN 多數決**，不用群心距離。HDBSCAN 的群在降維空間
   常是非凸的，群心指派會錯得很難察覺；kNN 在原始 1024 維 cosine 空間投票，
   非凸也不怕，而且票數與平均相似度天然就是信心指標。

4. **「遷移」拆三層。** 同一則職缺的描述不會變，硬做 job 層級的群間移動是假的：
   ① 群集的時間序列消長  ② 相鄰兩期的存活／汰換與重新定位  ③ 公司招募重心的位移。
"""

from __future__ import annotations

import argparse
import itertools
import json

import numpy as np
import pandas as pd
from sklearn.cluster import HDBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score

from jobshift import db
from jobshift import settings as S

GONE, ADDED = -999, -998   # 遷移矩陣的兩個虛擬端點：下架、新進


# ── 載入 ───────────────────────────────────────────────────────────
def available_snapshots() -> list[str]:
    """有向量檔的快照，依日期排序。這就是時間軸。"""
    snaps = sorted(p.stem for p in S.VEC_DIR.glob("*.npy"))
    if S.REFERENCE_SNAPSHOT not in snaps:
        raise SystemExit(
            f"[analyze] 找不到參考快照 {S.REFERENCE_SNAPSHOT} 的向量，請先跑 embed"
        )
    return [S.REFERENCE_SNAPSHOT] + [s for s in snaps if s != S.REFERENCE_SNAPSHOT]


def load(snapshot: str) -> tuple[pd.DataFrame, np.ndarray]:
    vecs = np.load(S.vec_npy(snapshot))
    ids = pd.read_parquet(S.vec_ids(snapshot))["job_id"].astype(str)
    with db.connect(read_only=True) as con:
        meta = con.execute(
            "SELECT job_id, job_title, company_id, company_name, industry_bucket, "
            "industry_name, city_name, salary_month_min, job_url "
            "FROM jobs WHERE snapshot_date = ?",
            [snapshot],
        ).df()
    meta["job_id"] = meta["job_id"].astype(str)
    meta = ids.to_frame().merge(meta, on="job_id", how="left")
    if len(meta) != len(vecs):
        raise SystemExit(f"{snapshot}: metadata {len(meta)} 筆 != 向量 {len(vecs)} 筆")
    return meta, vecs


# ── 參考座標系 ─────────────────────────────────────────────────────
def fit_reference(vecs: np.ndarray) -> tuple[np.ndarray, dict]:
    n = len(vecs)
    print(f"[analyze] 建參考座標系：{n} 筆 × {vecs.shape[1]} 維")

    pca = PCA(n_components=min(S.PCA_DIM, vecs.shape[1]), random_state=0).fit(vecs)
    low = pca.transform(vecs)
    print(
        f"[analyze] PCA {vecs.shape[1]}→{low.shape[1]} "
        f"（保留變異 {pca.explained_variance_ratio_.sum():.1%}）"
    )

    import umap

    emb = umap.UMAP(
        n_components=S.UMAP_DIM,
        n_neighbors=S.UMAP_NEIGHBORS,
        min_dist=0.0,
        metric="cosine",
        random_state=42,
    ).fit_transform(low)

    min_size = max(25, int(n * S.HDBSCAN_MIN_FRAC))
    labels = HDBSCAN(
        min_cluster_size=min_size, min_samples=10, cluster_selection_method="eom"
    ).fit_predict(emb)
    noise = float((labels == -1).mean())
    n_clu = len(set(labels)) - (1 if -1 in labels else 0)
    print(f"[analyze] HDBSCAN(min_cluster_size={min_size}) → {n_clu} 群，噪點 {noise:.1%}")

    method = f"hdbscan(min_cluster_size={min_size})"
    if noise > S.NOISE_FALLBACK or n_clu < 8:
        print(f"[analyze] 噪點超過 {S.NOISE_FALLBACK:.0%} 或群數過少 → 退回 KMeans")
        sub = emb[np.random.default_rng(0).choice(n, min(8000, n), replace=False)]
        best_k, best_s = S.KMEANS_K_GRID[0], -1.0
        for k in S.KMEANS_K_GRID:
            s = silhouette_score(sub, KMeans(n_clusters=k, n_init=4, random_state=0)
                                 .fit_predict(sub))
            print(f"           k={k:>3} silhouette={s:.4f}")
            if s > best_s:
                best_k, best_s = k, s
        labels = KMeans(n_clusters=best_k, n_init=10, random_state=0).fit_predict(emb)
        method = f"kmeans(k={best_k}, silhouette={best_s:.4f})"
        print(f"[analyze] 採用 {method}")

    n_clu = len(set(labels)) - (1 if -1 in labels else 0)
    return labels, {"method": method, "noise_ratio": noise, "n_reference_clusters": n_clu}


def discover(vecs: np.ndarray, min_frac: float = 0.02) -> np.ndarray:
    """在一小撮「指派不進參考池」的向量裡找結構。用於新型態偵測。"""
    n = len(vecs)
    low = PCA(n_components=min(S.PCA_DIM, n - 1, vecs.shape[1]), random_state=0).fit_transform(vecs)
    import umap

    emb = umap.UMAP(
        n_components=min(S.UMAP_DIM, n - 2),
        n_neighbors=min(S.UMAP_NEIGHBORS, n - 1),
        min_dist=0.0,
        metric="cosine",
        random_state=42,
    ).fit_transform(low)
    return HDBSCAN(min_cluster_size=max(25, int(n * min_frac)), min_samples=5).fit_predict(emb)


# ── 群集命名：不用斷詞，全靠 embedding + 字元 n-gram ─────────────────
def label_clusters(meta: pd.DataFrame, vecs: np.ndarray, labels: np.ndarray,
                   kind: str) -> pd.DataFrame:
    titles = np.array(meta["job_title"].fillna("").tolist())
    tfidf = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(2, 4),
        min_df=max(2, len(titles) // 5000), max_features=60000,
    )
    X = tfidf.fit_transform(titles.tolist())
    vocab = np.array(tfidf.get_feature_names_out())
    global_mean = np.asarray(X.mean(axis=0)).ravel()

    rows = []
    for cid in sorted(set(labels)):
        if cid == -1:
            continue
        mask = labels == cid
        # 代表職稱：離群心最近的真實職稱（比詞袋好讀，而且不需要 tokenizer）
        center = vecs[mask].mean(axis=0)
        center /= max(np.linalg.norm(center), 1e-9)
        top = np.argsort(-(vecs[mask] @ center))[:8]
        names = pd.Series(titles[mask][top]).drop_duplicates()
        # 鑑別詞：該群 tfidf 均值減全域均值，取最突出的字元 n-gram
        c_mean = np.asarray(X[mask].mean(axis=0)).ravel()
        terms = [t.strip() for t in vocab[np.argsort(-(c_mean - global_mean))[:12]]
                 if len(t.strip()) >= 2][:6]
        rows.append({
            "cluster_id": int(cid),
            "kind": kind,
            "label": names.iloc[0] if len(names) else f"群 {cid}",
            "top_titles": " / ".join(names.head(5)),
            "top_terms": " ".join(terms),
        })
    return pd.DataFrame(rows)


# ── 跨期指派：對參考池做 kNN 多數決 ─────────────────────────────────
def assign_by_knn(vecs: np.ndarray, pool_vecs: np.ndarray,
                  pool_labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    try:
        import torch

        dev = "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        torch, dev = None, "cpu"

    k = min(S.KNN_K, len(pool_vecs))
    n = len(vecs)
    out_lab = np.full(n, -1, dtype=np.int64)
    out_sim = np.zeros(n, dtype=np.float32)
    print(f"[analyze]   kNN 指派 {n} 筆 → 參考池 {len(pool_vecs)} 筆（k={k}, {dev}）")

    pool_t = torch.from_numpy(np.ascontiguousarray(pool_vecs)).to(dev) if torch else None
    for start in range(0, n, 1024):
        blk = np.ascontiguousarray(vecs[start:start + 1024])
        if torch is not None:
            sim = torch.from_numpy(blk).to(dev) @ pool_t.T
            top_sim_t, top_idx_t = torch.topk(sim, k, dim=1)
            top_sim, top_idx = top_sim_t.cpu().numpy(), top_idx_t.cpu().numpy()
        else:
            sim = blk @ pool_vecs.T
            top_idx = np.argpartition(-sim, k - 1, axis=1)[:, :k]
            top_sim = np.take_along_axis(sim, top_idx, axis=1)

        neigh = pool_labels[top_idx]
        for i in range(len(blk)):
            lab, sm = neigh[i], top_sim[i]
            valid = lab != -1
            if not valid.any():
                continue
            vals, counts = np.unique(lab[valid], return_counts=True)
            win = vals[counts.argmax()]
            mean_sim = float(sm[lab == win].mean())
            out_sim[start + i] = mean_sim
            if counts.max() / k >= S.KNN_VOTE_RATIO and mean_sim >= S.KNN_SIM_FLOOR:
                out_lab[start + i] = win
    hit = (out_lab != -1).mean()
    print(f"[analyze]   指派成功率 {hit:.1%}，未歸類 {(out_lab == -1).sum()} 筆")
    return out_lab, out_sim


# ── 相鄰兩期的遷移 ─────────────────────────────────────────────────
def pairwise(prev: str, cur: str, a: pd.DataFrame, b: pd.DataFrame,
             va: np.ndarray, vb: np.ndarray) -> dict[str, pd.DataFrame]:
    ids_a, ids_b = set(a.job_id), set(b.job_id)
    kept = sorted(ids_a & ids_b)

    # 存活／汰換（依產業）
    surv = []
    for bucket in sorted(set(a.industry_bucket) | set(b.industry_bucket)):
        o = set(a.loc[a.industry_bucket == bucket, "job_id"])
        n = set(b.loc[b.industry_bucket == bucket, "job_id"])
        keep = len(o & n)
        surv.append({
            "from_snapshot": prev, "to_snapshot": cur, "industry_bucket": bucket,
            "n_from": len(o), "n_to": len(n), "n_survived": keep,
            "n_gone": len(o) - keep, "n_added": len(n) - keep,
            "churn_rate": (len(o) - keep) / len(o) if o else np.nan,
            "renewal_rate": (len(n) - keep) / len(n) if n else np.nan,
        })

    # 遷移矩陣：存活職缺的 前期群集 → 後期群集，加上下架／新進兩個端點
    ca = a.set_index("job_id").cluster_id
    cb = b.set_index("job_id").cluster_id
    pair = pd.DataFrame({"from_cluster": ca.reindex(kept).to_numpy(),
                         "to_cluster": cb.reindex(kept).to_numpy()})
    mat = pair.groupby(["from_cluster", "to_cluster"]).size().reset_index(name="n")
    mat["kind"] = np.where(mat.from_cluster == mat.to_cluster, "stayed", "moved")
    gone = (a[~a.job_id.isin(kept)].groupby("cluster_id").size().reset_index(name="n")
            .rename(columns={"cluster_id": "from_cluster"}).assign(to_cluster=GONE, kind="gone"))
    added = (b[~b.job_id.isin(kept)].groupby("cluster_id").size().reset_index(name="n")
             .rename(columns={"cluster_id": "to_cluster"}).assign(from_cluster=ADDED, kind="added"))
    flow = pd.concat([mat, gone, added], ignore_index=True)
    flow["from_snapshot"], flow["to_snapshot"] = prev, cur
    flow = flow[["from_snapshot", "to_snapshot", "from_cluster", "to_cluster", "n", "kind"]]

    # 公司招募重心的位移
    shift = []
    idx_a = {c: g.index.to_numpy() for c, g in a.groupby("company_id")}
    for comp, gb in b.groupby("company_id"):
        ga = idx_a.get(comp)
        if ga is None or len(ga) < 3 or len(gb) < 3:
            continue
        vo, vn = va[ga].mean(axis=0), vb[gb.index.to_numpy()].mean(axis=0)
        cos = float(vo @ vn / max(np.linalg.norm(vo) * np.linalg.norm(vn), 1e-9))
        fo = a.loc[ga].cluster_id
        fo, fn = fo[fo != -1], gb.cluster_id[gb.cluster_id != -1]
        shift.append({
            "from_snapshot": prev, "to_snapshot": cur, "company_id": comp,
            "company_name": gb.company_name.iloc[0],
            "industry_bucket": gb.industry_bucket.iloc[0],
            "n_from": len(ga), "n_to": len(gb),
            "cos_similarity": cos, "shift_score": 1 - cos,
            "from_cluster": int(fo.mode().iloc[0]) if len(fo) else -1,
            "to_cluster": int(fn.mode().iloc[0]) if len(fn) else -1,
        })

    gone_n = int(flow.loc[flow.kind == "gone", "n"].sum())
    added_n = int(flow.loc[flow.kind == "added", "n"].sum())
    print(f"[analyze] {prev} → {cur}：存活 {len(kept)}/{len(ids_a)} "
          f"（{len(kept) / max(len(ids_a), 1):.1%}）、下架 {gone_n}、新進 {added_n}；"
          f"公司位移樣本 {len(shift)} 家")
    return {"survival": pd.DataFrame(surv), "cluster_flow": flow,
            "company_shift": pd.DataFrame(shift)}


# ── 主流程 ─────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser(description="對所有已向量化的快照做分群與遷移分析")
    ap.add_argument("--refit", action="store_true",
                    help="重建參考座標系（會讓既有群集編號全部作廢）")
    args = ap.parse_args()

    snaps = available_snapshots()
    print(f"[analyze] 時間軸：{' → '.join(snaps)}")

    ref_meta, ref_vecs = load(S.REFERENCE_SNAPSHOT)
    with db.connect(read_only=True) as con:
        cached = (db.table_exists(con, "reference_labels") and not args.refit
                  and con.execute("SELECT count(*) FROM reference_labels").fetchone()[0]
                  == len(ref_meta))
        ref_labels_df = con.execute("SELECT * FROM reference_labels").df() if cached else None

    if cached:
        ref_labels = (ref_labels_df.set_index("job_id").cluster_id
                      .reindex(ref_meta.job_id).to_numpy())
        info = {"method": "cached", "noise_ratio": float((ref_labels == -1).mean()),
                "n_reference_clusters": int(len(set(ref_labels)) - 1)}
        print(f"[analyze] 沿用既有參考座標系（{info['n_reference_clusters']} 群）"
              "；要重建請加 --refit")
    else:
        ref_labels, info = fit_reference(ref_vecs)

    catalog = label_clusters(ref_meta, ref_vecs, ref_labels, "reference")
    pool_vecs, pool_labels = ref_vecs, ref_labels
    next_id = int(catalog.cluster_id.max()) + 1 if len(catalog) else 0

    assigned: dict[str, tuple[pd.DataFrame, np.ndarray]] = {
        S.REFERENCE_SNAPSHOT: (ref_meta.assign(cluster_id=ref_labels, sim=1.0), ref_vecs)
    }

    stability: list[dict] = []
    for i, snap in enumerate(snaps[1:], start=1):
        print(f"\n[analyze] === {snap} ===")
        meta, vecs = load(snap)
        lab, sim = assign_by_knn(vecs, pool_vecs, pool_labels)

        # 存活職缺直接繼承上一期的群集編號，不重新指派。
        # 理由：同一則職缺的描述兩期通常一模一樣，向量也一樣 —— 它「換群」只可能是
        # 指派方法的抖動（參考期標籤來自 HDBSCAN，新期來自 kNN 投票，兩套機制對同一個點
        # 可以給不同答案），不是市場的變化。不繼承的話遷移矩陣會被這種假訊號灌滿。
        prev = snaps[i - 1]
        prev_lab = assigned[prev][0].set_index("job_id").cluster_id
        surv = meta.job_id.isin(prev_lab.index).to_numpy()
        if surv.any():
            inherited = prev_lab.reindex(meta.job_id[surv]).to_numpy()
            # 繼承之前先量一次「重新指派會有多少不一致」—— 這就是本方法的雜訊底線，
            # 拿它當誤差棒，才知道別的變化有沒有超出雜訊。
            both = (lab[surv] != -1) & (inherited != -1)
            agree = float((lab[surv][both] == inherited[both]).mean()) if both.any() else np.nan
            stability.append({
                "from_snapshot": prev, "to_snapshot": snap,
                "n_survivors": int(surv.sum()), "n_comparable": int(both.sum()),
                "agreement": agree, "noise_floor": 1 - agree,
            })
            print(f"[analyze]   指派穩定度：存活 {surv.sum()} 筆中可比較 {both.sum()} 筆，"
                  f"重新指派一致率 {agree:.1%}（＝雜訊底線 {1 - agree:.1%}）")
            lab[surv] = inherited

        # 指派不進去的 → 獨立分群 → 成為新群集 → 併回參考池，下一期就能比對到
        idx = np.where(lab == -1)[0]
        if len(idx) >= 60:
            sub = discover(vecs[idx])
            n_new = len(set(sub)) - (1 if -1 in sub else 0)
            print(f"[analyze]   新型態偵測：{len(idx)} 筆未歸類 → {n_new} 個新群集")
            if n_new:
                remap = {old: next_id + i
                         for i, old in enumerate(sorted(c for c in set(sub) if c != -1))}
                new_rows = label_clusters(meta.iloc[idx].reset_index(drop=True),
                                          vecs[idx], sub, "emergent")
                new_rows["cluster_id"] = new_rows.cluster_id.map(remap)
                new_rows["first_seen"] = snap
                catalog = pd.concat([catalog, new_rows], ignore_index=True)
                lab[idx] = [remap.get(c, -1) for c in sub]
                grew = idx[sub != -1]
                pool_vecs = np.vstack([pool_vecs, vecs[grew]])
                pool_labels = np.concatenate([pool_labels, lab[grew]])
                next_id += n_new
                print(f"[analyze]   參考池成長至 {len(pool_vecs)} 筆")
        assigned[snap] = (meta.assign(cluster_id=lab, sim=sim), vecs)

    # 時間序列
    ts = []
    for snap, (meta, _) in assigned.items():
        vc = meta.cluster_id.value_counts()
        for cid, n in vc.items():
            ts.append({"snapshot_date": snap, "cluster_id": int(cid), "n": int(n),
                       "share": n / len(meta)})
    ts_df = pd.DataFrame(ts)

    ind = []
    for snap, (meta, _) in assigned.items():
        g = (meta[meta.cluster_id != -1].groupby(["industry_bucket", "cluster_id"])
             .size().reset_index(name="n"))
        g["snapshot_date"] = snap
        g["share_in_industry"] = g.n / g.groupby("industry_bucket").n.transform("sum")
        ind.append(g)

    # 相鄰兩期的遷移
    pairs = {"survival": [], "cluster_flow": [], "company_shift": []}
    for prev, cur in itertools.pairwise(snaps):
        (a, va), (b, vb) = assigned[prev], assigned[cur]
        for key, dfp in pairwise(prev, cur, a, b, va, vb).items():
            pairs[key].append(dfp)

    if "first_seen" not in catalog.columns:
        catalog["first_seen"] = S.REFERENCE_SNAPSHOT
    catalog["first_seen"] = catalog.first_seen.fillna(S.REFERENCE_SNAPSHOT)

    meta_rows = pd.DataFrame([
        {"key": "reference_snapshot", "value": S.REFERENCE_SNAPSHOT},
        {"key": "snapshots", "value": json.dumps(snaps)},
        {"key": "latest_snapshot", "value": snaps[-1]},
        {"key": "cluster_method", "value": info["method"]},
        {"key": "noise_ratio", "value": f"{info['noise_ratio']:.4f}"},
        {"key": "n_reference_clusters", "value": str(info["n_reference_clusters"])},
        {"key": "n_clusters_total", "value": str(len(catalog))},
        {"key": "params", "value": json.dumps(
            {"knn_k": S.KNN_K, "vote_ratio": S.KNN_VOTE_RATIO,
             "sim_floor": S.KNN_SIM_FLOOR, "umap_dim": S.UMAP_DIM}, ensure_ascii=False)},
    ])

    jc = pd.concat([m.assign(snapshot_date=s)[["snapshot_date", "job_id", "cluster_id", "sim"]]
                    for s, (m, _) in assigned.items()], ignore_index=True)

    with db.connect() as con:
        db.replace_table(con, "reference_labels",
                         pd.DataFrame({"job_id": ref_meta.job_id, "cluster_id": ref_labels}))
        db.replace_table(con, "clusters", catalog)
        db.replace_table(con, "cluster_timeseries", ts_df)
        db.replace_table(con, "flow_industry_cluster", pd.concat(ind, ignore_index=True))
        db.replace_table(con, "job_cluster", jc)
        for key, frames in pairs.items():
            db.replace_table(con, key, pd.concat(frames, ignore_index=True)
                             if frames else pd.DataFrame())
        db.replace_table(con, "assignment_stability", pd.DataFrame(stability))
        db.replace_table(con, "analysis_meta", meta_rows)

    print(f"\n[analyze] 完成。快照 {len(snaps)} 份、群集 {len(catalog)} 個"
          f"（其中新生 {int((catalog.kind == 'emergent').sum())} 個）")


if __name__ == "__main__":
    main()

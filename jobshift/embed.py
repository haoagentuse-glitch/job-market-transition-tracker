"""bge-m3 向量化（GPU）。輸出 float32、已 L2 正規化的 .npy ＋ 對照用 ids.parquet。

向量存 .npy 而不是塞進 DuckDB：後面分群與 kNN 全是稠密矩陣運算，
numpy/torch 直接吃記憶體映射最快；DuckDB 只存 metadata 與分析結果。
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

from jobshift import db
from jobshift import settings as S


def embed_snapshot(snapshot: str, limit: int | None = None, force: bool = False) -> None:
    npy_path, ids_path = S.vec_npy(snapshot), S.vec_ids(snapshot)
    if npy_path.exists() and not force:
        print(f"[embed] {snapshot} 已存在 {npy_path.name}，跳過（--force 可覆寫）")
        return

    with db.connect(read_only=True) as con:
        sql = ("SELECT job_id, text_for_embed FROM jobs "
               "WHERE snapshot_date = ? ORDER BY job_id")
        if limit:
            sql += f" LIMIT {int(limit)}"
        df = con.execute(sql, [snapshot]).df()
    if df.empty:
        raise SystemExit(f"[embed] jobs 表裡沒有 {snapshot} 的資料，請先跑 ingest")

    import torch
    from sentence_transformers import SentenceTransformer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[embed] {snapshot}: {len(df)} 筆 | device={device} | model={S.EMBED_MODEL}")
    if device == "cpu":
        print("[embed] 警告：沒吃到 GPU，6.5 萬筆在 CPU 上要數小時。"
              "請確認 compose 的 deploy.resources 生效。")

    model = SentenceTransformer(S.EMBED_MODEL, device=device)
    model.max_seq_length = S.EMBED_MAX_LEN
    if device == "cuda":
        model = model.half()

    t0 = time.time()
    vecs = model.encode(
        df["text_for_embed"].fillna("").tolist(),
        batch_size=S.EMBED_BATCH,
        normalize_embeddings=True,      # 之後全部用內積當 cosine
        show_progress_bar=True,
        convert_to_numpy=True,
    ).astype(np.float32)

    np.save(npy_path, vecs)
    pd.DataFrame({"job_id": df["job_id"].astype(str)}).to_parquet(ids_path, index=False)
    dt = time.time() - t0
    print(f"[embed] 完成 shape={vecs.shape} 耗時 {dt:.0f}s "
          f"({len(df) / max(dt, 1):.0f} 筆/秒) → {npy_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help="舊快照 + 新快照都跑")
    ap.add_argument("--snapshot", default=S.SNAPSHOT_DATE)
    ap.add_argument("--limit", type=int, default=None, help="只取前 N 筆（骨架驗證用）")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    targets = [S.BASE_SNAPSHOT, args.snapshot] if args.all else [args.snapshot]
    for snap in targets:
        embed_snapshot(snap, limit=args.limit, force=args.force)


if __name__ == "__main__":
    main()
